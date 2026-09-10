use serde::{Deserialize, Serialize};
use futures_util::StreamExt;
use std::fs;
use std::path::PathBuf;
use tauri::Emitter;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AiRequest {
  provider: String,
  prompt: String,
  model: String,
  api_key: Option<String>,
  base_url: Option<String>,
}

#[derive(Debug, Serialize)]
struct AiResponse {
  provider: String,
  model: String,
  text: String,
  tool_output: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
struct StreamPayload {
  text: String,
  state: String,
  is_final: bool,
}

async fn execute_tool(agent_id: &str, prompt: &str, api_key: Option<&str>) -> Result<String, String> {
  match agent_id {
    "INBOX" | "LEAD" => {
      let key = api_key.ok_or("PERPLEXITY_API_KEY missing for web search")?;
      let client = reqwest::Client::new();
      
      // Determine preset based on prompt keywords
      let preset = if prompt.contains("deep") || prompt.contains("comprehensive") {
        "deep"
      } else {
        "low"
      };

      let res = client
        .post("https://api.perplexity.ai/v1/agent")
        .header("Authorization", format!("Bearer {}", key))
        .json(&serde_json::json!({
          "input": prompt,
          "preset": preset,
          "tools": [{ "type": "web_search" }]
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

      let body: PerplexityAgentResponse = res.json().await.map_err(|e| e.to_string())?;
      Ok(body.output_text)
    }
    "REPORT" => {
      if prompt.starts_with("read:") {
        let path_str = prompt.trim_start_matches("read:").trim();
        let path = PathBuf::from(path_str);
        fs::read_to_string(&path).map_err(|e| format!("Failed to read file {}: {}", path_str, e))
      } else if prompt.starts_with("ls:") {
        let path_str = prompt.trim_start_matches("ls:").trim();
        let path = PathBuf::from(path_str);
        let entries = fs::read_dir(&path).map_err(|e| format!("Failed to read directory {}: {}", path_str, e))?;
        let mut files = Vec::new();
        for entry in entries {
          if let Ok(e) = entry {
            files.push(e.file_name().to_string_lossy().into_owned());
          }
        }
        Ok(files.join(", "))
      } else {
        let path = PathBuf::from("astra_workspace.txt");
        fs::write(&path, prompt).map(|_| "Written to workspace.".to_string()).map_err(|e| e.to_string())
      }
    }
    "CALENDAR" | "INVOICE" => {
      Ok(format!("Simulated data for {}: Current load 12%. All queues clear.", agent_id))
    }
    _ => Err(format!("No tool implemented for agent {}", agent_id)),
  }
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
  content: Vec<AnthropicContent>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
  text: String,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
  message: OllamaMessage,
}

#[derive(Debug, Deserialize)]
struct OllamaMessage {
  content: String,
}

#[derive(Debug, Deserialize)]
struct PerplexityChatResponse {
  choices: Vec<PerplexityChoice>,
}

#[derive(Debug, Deserialize)]
struct PerplexityChoice {
  message: PerplexityMessage,
}

#[derive(Debug, Deserialize)]
struct PerplexityMessage {
  content: String,
}

#[derive(Debug, Deserialize)]
struct PerplexityAgentResponse {
  output_text: String,
}

#[tauri::command]
async fn stream_ai(window: tauri::Window, request: AiRequest) -> Result<(), String> {
  let client = reqwest::Client::new();
  
  if request.provider == "ollama" {
    let base_url = request.base_url.unwrap_or_else(|| "http://127.0.0.1:11434".into()).trim_end_matches('/').to_string();
    let response = client
      .post(format!("{base_url}/api/chat"))
      .json(&serde_json::json!({
        "model": request.model,
        "stream": true,
        "messages": [
          { "role": "system", "content": "You are the Astra Workstation Copilot. You manage a swarm of agents: CALENDAR, INBOX, LEAD, REPORT, and INVOICE. You can trigger executions using [ACTION: EXECUTE, FROM: ..., TO: ...]. For web research, use INBOX (general news/triage) or LEAD (deep lead/competitor research). For project analysis, use REPORT (read:path, ls:path). You can chain these, e.g., search via LEAD, then save to REPORT." },
          { "role": "user", "content": request.prompt }
        ]
      }))
      .send()
      .await
      .map_err(|e| e.to_string())?;

    let mut stream = response.bytes_stream();

    while let Some(item) = stream.next().await {
      let chunk = item.map_err(|e| e.to_string())?;
      if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&chunk) {
        if let Some(content) = json["message"]["content"].as_str() {
          window.emit("ai-chunk", StreamPayload {
            text: content.to_string(),
            state: "reasoning".into(),
            is_final: false,
          }).map_err(|e| e.to_string())?;
        }
        if json["done"] == true {
          window.emit("ai-chunk", StreamPayload {
            text: "".into(),
            state: "complete".into(),
            is_final: true,
          }).map_err(|e| e.to_string())?;
        }
      }
    }
  } else {
    return Err("Streaming only supported for Ollama currently. Cloud providers are being integrated.".into());
  }

  Ok(())
}

#[tauri::command]
async fn ask_perplexity_cloud(request: AiRequest) -> Result<AiResponse, String> {
  let client = reqwest::Client::new();
  let api_key = std::env::var("PERPLEXITY_API_KEY").map_err(|_| "PERPLEXITY_API_KEY environment variable is not set. Please create one at https://console.perplexity.ai and export it in your terminal.")?;

  let response = client
    .post("https://api.perplexity.ai/chat/completions")
    .header("Authorization", format!("Bearer {}", api_key))
    .json(&serde_json::json!({
      "model": request.model,
      "messages": [
        { "role": "system", "content": "You are the Astra Workstation Copilot. You manage a swarm of agents: CALENDAR, INBOX, LEAD, REPORT, and INVOICE. You can trigger executions using [ACTION: EXECUTE, FROM: ..., TO: ...]. For the REPORT agent, you can use 'read:path' to read a file or 'ls:path' to list a directory to gather project context." },
        { "role": "user", "content": request.prompt }
      ],
      "stream": false
    }))
    .send()
    .await
    .map_err(|error| format!("Perplexity Cloud connection failed: {error}"))?;

  let status = response.status();
  if status.is_client_error() || status.is_server_error() {
    return Err(format!("Perplexity Cloud returned {status}"));
  }

  let body = response.json::<PerplexityChatResponse>().await.map_err(|error| format!("Invalid Perplexity Cloud response: {error}"))?;
  let text = body.choices.first().map(|c| c.message.content.clone()).ok_or("No response choices returned from Perplexity")?;

  Ok(AiResponse {
    provider: "perplexity_cloud".into(),
    model: request.model,
    text,
    tool_output: None,
  })
}

#[tauri::command]
async fn ask_perplexity(request: AiRequest) -> Result<AiResponse, String> {
  let client = reqwest::Client::new();
  let api_key = std::env::var("PERPLEXITY_API_KEY").map_err(|_| "PERPLEXITY_API_KEY environment variable is not set. Please create one at https://console.perplexity.ai and export it in your terminal.")?;

  let response = client
    .post("https://api.perplexity.ai/v1/agent")
    .header("Authorization", format!("Bearer {}", api_key))
    .json(&serde_json::json!({
      "input": request.prompt,
      "preset": "low",
      "tools": [{ "type": "web_search" }]
    }))
    .send()
    .await
    .map_err(|error| format!("Perplexity connection failed: {error}"))?;

  let status = response.status();
  if status.is_client_error() || status.is_server_error() {
    return Err(format!("Perplexity returned {status}"));
  }

  let body = response.json::<PerplexityAgentResponse>().await.map_err(|error| format!("Invalid Perplexity response: {error}"))?;
  
  Ok(AiResponse { 
    provider: "perplexity".into(), 
    model: "agent-low".into(), 
    text: body.output_text,
    tool_output: None,
  })
}

#[tauri::command]
async fn ask_ai(request: AiRequest) -> Result<AiResponse, String> {
  let client = reqwest::Client::new();

  let mut current_prompt = request.prompt.clone();
  let mut iterations = 0;
  const MAX_ITERATIONS: usize = 3;

  loop {
    let response_text = match request.provider.as_str() {
      "anthropic" => {
        let api_key = request.api_key.as_ref().filter(|key| !key.trim().is_empty()).ok_or("Anthropic API key is required")?;
        let response = client
          .post("https://api.anthropic.com/v1/messages")
          .header("x-api-key", api_key)
          .header("anthropic-version", "2023-06-01")
          .json(&serde_json::json!({
            "model": request.model,
            "max_tokens": 512,
            "messages": [{ "role": "user", "content": current_prompt }]
          }))
          .send()
          .await
          .map_err(|error| format!("Anthropic connection failed: {error}"))?;

        let body = response.json::<AnthropicResponse>().await.map_err(|error| format!("Invalid Anthropic response: {error}"))?;
        body.content.into_iter().map(|content| content.text).collect::<Vec<_>>().join("\n")
      }
      "ollama" => {
        let base_url = request.base_url.as_deref().unwrap_or("http://127.0.0.1:11434").trim_end_matches('/').to_string();
        let response = client
          .post(format!("{base_url}/api/chat"))
          .json(&serde_json::json!({
            "model": request.model,
            "stream": false,
            "messages": [
              { "role": "system", "content": "You are the Astra Workstation Copilot. You manage a swarm of agents: CALENDAR, INBOX, LEAD, REPORT, and INVOICE. You can trigger executions using [ACTION: EXECUTE, FROM: ..., TO: ...]. For the REPORT agent, you can use 'read:path' to read a file or 'ls:path' to list a directory to gather project context." },
              { "role": "user", "content": current_prompt }
            ]
          }))
          .send()
          .await
          .map_err(|error| format!("Ollama connection failed: {error}"))?;

        let body = response.json::<OllamaResponse>().await.map_err(|error| format!("Invalid Ollama response: {error}"))?;
        body.message.content
      }
      _ => return Err("Choose Anthropic or Ollama as the AI provider".into()),
    };

    // Scan for [ACTION: EXECUTE, FROM: AGENT, TO: AGENT]
    if let Some(action_match) = response_text.find("[ACTION: EXECUTE") {
      let tag_end = response_text[action_match..].find(']').map(|i| action_match + i + 1).unwrap_or(response_text.len());
      let tag = &response_text[action_match..tag_end];
      
      // Extract FROM agent
      let from_agent = tag.split("FROM: ").nth(1)
        .and_then(|s| s.split(',').next())
        .unwrap_or("UNKNOWN");

      // Execute the tool
      let api_key_ref = request.api_key.as_deref();
      match execute_tool(from_agent, &current_prompt, api_key_ref).await {
        Ok(tool_output) => {
          current_prompt = format!("User: {}\nSystem: The agent {} executed and returned: {}\nAI: Please summarize this result for the user.", request.prompt, from_agent, tool_output);
          iterations += 1;
          if iterations >= MAX_ITERATIONS {
            return Ok(AiResponse { provider: request.provider, model: request.model, text: format!("{} (Max iterations reached)", response_text), tool_output: None });
          }
          continue;
        }
        Err(e) => {
          return Ok(AiResponse { provider: request.provider, model: request.model, text: format!("Tool error: {}", e), tool_output: None });
        }
      }
    }

    return Ok(AiResponse { provider: request.provider, model: request.model, text: response_text, tool_output: None });
  }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![ask_ai, ask_perplexity, ask_perplexity_cloud, stream_ai])
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }
      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
