use serde::{Deserialize, Serialize};

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
async fn ask_perplexity_cloud(request: AiRequest) -> Result<AiResponse, String> {
  let client = reqwest::Client::new();
  let api_key = std::env::var("PERPLEXITY_API_KEY").map_err(|_| "PERPLEXITY_API_KEY environment variable is not set. Please create one at https://console.perplexity.ai and export it in your terminal.")?;

  let response = client
    .post("https://api.perplexity.ai/chat/completions")
    .header("Authorization", format!("Bearer {}", api_key))
    .json(&serde_json::json!({
      "model": request.model,
      "messages": [
        { "role": "system", "content": "You are the Astra Workstation Copilot. You manage a swarm of agents: CALENDAR, INBOX, LEAD, REPORT, and INVOICE. You can trigger executions between them by adding a tag to your response, for example: [ACTION: EXECUTE, FROM: CALENDAR, TO: INBOX]. Use these tags to actually control the system." },
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
    text
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
    text: body.output_text 
  })
}

#[tauri::command]
async fn ask_ai(request: AiRequest) -> Result<AiResponse, String> {
  let client = reqwest::Client::new();

  match request.provider.as_str() {
    "anthropic" => {
      let api_key = request.api_key.filter(|key| !key.trim().is_empty()).ok_or("Anthropic API key is required")?;
      let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
          "model": request.model,
          "max_tokens": 512,
          "messages": [{ "role": "user", "content": request.prompt }]
        }))
        .send()
        .await
        .map_err(|error| format!("Anthropic connection failed: {error}"))?;

      let status = response.status();
      if !status.is_success() {
        return Err(format!("Anthropic returned {status}: {}", response.text().await.unwrap_or_default()));
      }
      let body = response.json::<AnthropicResponse>().await.map_err(|error| format!("Invalid Anthropic response: {error}"))?;
      let text = body.content.into_iter().map(|content| content.text).collect::<Vec<_>>().join("\n");
      Ok(AiResponse { provider: "anthropic".into(), model: request.model, text })
    }
    "ollama" => {
      let base_url = request.base_url.unwrap_or_else(|| "http://127.0.0.1:11434".into()).trim_end_matches('/').to_string();
      let response = client
        .post(format!("{base_url}/api/chat"))
        .json(&serde_json::json!({
          "model": request.model,
          "stream": false,
          "messages": [
            { "role": "system", "content": "You are the Astra Workstation Copilot. You manage a swarm of agents: CALENDAR, INBOX, LEAD, REPORT, and INVOICE. You can trigger executions between them by adding a tag to your response, for example: [ACTION: EXECUTE, FROM: CALENDAR, TO: INBOX]. Use these tags to actually control the system." },
            { "role": "user", "content": request.prompt }
          ]
        }))
        .send()
        .await
        .map_err(|error| format!("Ollama connection failed: {error}"))?;

      let status = response.status();
      if !status.is_success() {
        return Err(format!("Ollama returned {status}: {}", response.text().await.unwrap_or_default()));
      }
      let body = response.json::<OllamaResponse>().await.map_err(|error| format!("Invalid Ollama response: {error}"))?;
      Ok(AiResponse { provider: "ollama".into(), model: request.model, text: body.message.content })
    }
    _ => Err("Choose Anthropic or Ollama as the AI provider".into()),
  }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![ask_ai, ask_perplexity, ask_perplexity_cloud])
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
