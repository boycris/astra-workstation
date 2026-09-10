import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { EffectComposer, Bloom, ChromaticAberration, Noise, SMAA, ToneMapping, Vignette, wrapEffect } from "@react-three/postprocessing";
import { BlendFunction, ToneMappingMode } from "postprocessing";
import { Float, OrbitControls, Stars, Text, Environment } from "@react-three/drei";
import { createNoise3D } from "simplex-noise";
import * as THREE from "three";
import { ColorGradeEffect } from "./scene/effects/ColorGradeEffect";
import "./styles.css";

type Agent = {
  id: string;
  name: string;
  sub: string;
  pos: [number, number, number];
  colorIdle: string;
  colorActive: string;
  dynamicLabel?: string;
};

type CopilotMessage = {
  role: "user" | "ai";
  text: string;
};

type AiProvider = "ollama" | "anthropic" | "perplexity" | "perplexity_cloud";
type CoreState = "idle" | "listening" | "searching" | "reasoning" | "tool_use" | "error" | "complete";
type Quality = "low" | "medium" | "ultra";

type StreamPayload = {
  text: string;
  state?: string;
  is_final?: boolean;
};

const CLOUD_MODELS = {
  "DeepSeek Reasoner": "deepseek-reasoner",
  "Claude 3.5 Sonnet": "claude-3-5-sonnet",
  "GPT-4o Mini": "gpt-4o-mini",
  "Llama 3.1 405B": "llama-3.1-405b",
};

const NEBULA_VERTEX_SHADER = `
  varying vec3 vWorldPosition;
  varying vec3 vNormal;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPosition.xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const NEBULA_FRAGMENT_SHADER = `
  uniform float uTime;
  uniform vec3 uColor;
  uniform float uIntensity;
  varying vec3 vWorldPosition;
  varying vec3 vNormal;

  float hash(float n) { return fract(sin(n) * 43758.5453123); }
  float noise(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0 + 113.0*p.z;
    return mix(mix(mix(hash(n+0.0), hash(n+1.0), f.x),
                   mix(hash(n+57.0), hash(n+58.0), f.x), f.y),
               mix(mix(hash(n+113.0), hash(n+114.0), f.x),
                   mix(hash(n+170.0), hash(n+171.0), f.x), f.y), f.z);
  }

  float fbm(vec3 p) {
    float f = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 6; i++) {
      f += amp * noise(p);
      p *= 2.0;
      amp *= 0.5;
    }
    return f;
  }

  void main() {
    vec3 viewDir = normalize(cameraPosition - vWorldPosition);
    float density = 0.0;
    vec3 p = vWorldPosition * 0.4;
    
    for(int i = 0; i < 12; i++) {
      p += viewDir * 0.12;
      density += fbm(p + uTime * 0.15);
    }

    float finalDensity = density * uIntensity * 0.15;
    
    // Rim lighting effect
    float rim = 1.0 - max(dot(vNormal, -viewDir), 0.0);
    rim = pow(rim, 3.0) * 0.5;
    
    vec3 finalColor = uColor * finalDensity * (1.0 + sin(uTime * 1.5) * 0.15) + (uColor * rim);
    
    gl_FragColor = vec4(finalColor, finalDensity * 0.7);
  }
`;

const noise3D = createNoise3D();
const ColorGrade = wrapEffect(ColorGradeEffect as any) as any;

const AGENTS: Agent[] = [
  { id: "CALENDAR", name: "calendar sync", sub: "1.25M actions / syncpool", pos: [-5.8, 4.0, -1.5], colorIdle: "#00f0ff", colorActive: "#ff2a85" },
  { id: "INBOX", name: "inbox triage", sub: "4.89M actions / parser", pos: [5.9, 4.4, 1.2], colorIdle: "#38bdf8", colorActive: "#f43f5e" },
  { id: "LEAD", name: "lead scrape", sub: "348k actions / headless", pos: [6.4, -1.0, -2], colorIdle: "#a855f7", colorActive: "#ff0055" },
  { id: "REPORT", name: "report build", sub: "2.11M actions / summary", pos: [-5.3, -3.7, 0.8], colorIdle: "#06b6d4", colorActive: "#ec4899" },
  { id: "INVOICE", name: "invoice run", sub: "890k actions / billing", pos: [2.4, -4.1, 2.2], colorIdle: "#818cf8", colorActive: "#e11d48" },
];

function AgentFixture({ agent, isActive }: { agent: Agent; isActive: boolean }) {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (!groupRef.current) return;
    const t = state.clock.elapsedTime;
    groupRef.current.rotation.y = Math.sin(t * 0.2) * 0.1;
    groupRef.current.position.y = Math.sin(t * 0.5) * 0.1;
  });

  return (
    <group ref={groupRef} position={agent.pos}>
      <mesh>
        <icosahedronGeometry args={[0.72, 1]} />
        <meshStandardMaterial color={isActive ? agent.colorActive : agent.colorIdle} emissive={isActive ? agent.colorActive : "#000000"} emissiveIntensity={isActive ? 2 : 0.2} wireframe />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.92, 0.025, 8, 48]} />
        <meshBasicMaterial color={isActive ? agent.colorActive : agent.colorIdle} transparent opacity={0.7} />
      </mesh>
      {isActive && agent.dynamicLabel && (
        <Text position={[0, 2.5, 0]} fontSize={0.28} color="#ffffff" anchorX="center" anchorY="middle">
          {agent.dynamicLabel}
        </Text>
      )}
    </group>
  );
}

function AgentCloud({ agent, isActive, isTarget, quality }: { agent: Agent; isActive: boolean; isTarget: boolean; quality: Quality }) {
  const pointsRef = useRef<THREE.Points>(null);
  const count = quality === "ultra" ? 900 : quality === "medium" ? 400 : 150;
  const [positions, basePositions] = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const base = new Float32Array(count * 3);
    for (let index = 0; index < count; index++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const radius = Math.cbrt(Math.random()) * 1.6;
      const x = radius * Math.sin(phi) * Math.cos(theta);
      const y = radius * Math.sin(phi) * Math.sin(theta);
      const z = radius * Math.cos(phi);
      positions.set([x, y, z], index * 3);
      base.set([x, y, z], index * 3);
    }
    return [positions, base];
  }, [count]);
  const linePositions = useMemo(() => {
    const lines = new Float32Array(180 * 6);
    for (let index = 0; index < 180; index++) {
      const a = Math.floor(Math.random() * count);
      const b = Math.floor(Math.random() * count);
      lines.set([...basePositions.slice(a * 3, a * 3 + 3), ...basePositions.slice(b * 3, b * 3 + 3)], index * 6);
    }
    return lines;
  }, [basePositions, count]);

  useFrame((state) => {
    if (!pointsRef.current) return;
    const time = state.clock.elapsedTime * (isActive ? 2.5 : 0.6);
    const array = pointsRef.current.geometry.attributes.position.array as Float32Array;
    for (let index = 0; index < count; index++) {
      const offset = index * 3;
      const noise = noise3D(basePositions[offset] * 0.8 + time * 0.2, basePositions[offset + 1] * 0.8 + time * 0.2, basePositions[offset + 2] * 0.8 + time * 0.2);
      const scale = 1 + (isActive ? 0.35 : 0.1) * noise;
      array[offset] = basePositions[offset] * scale;
      array[offset + 1] = basePositions[offset + 1] * scale;
      array[offset + 2] = basePositions[offset + 2] * scale;
    }
    pointsRef.current.geometry.attributes.position.needsUpdate = true;
  });

  const color = isActive ? agent.colorActive : isTarget ? "#ffffff" : agent.colorIdle;
  return (
    <group position={agent.pos}>
      <points ref={pointsRef}>
        <bufferGeometry><bufferAttribute attach="attributes-position" args={[positions, 3]} /></bufferGeometry>
        <pointsMaterial size={isActive ? 0.055 : 0.035} color={color} transparent opacity={isActive ? 0.95 : 0.65} blending={THREE.AdditiveBlending} depthWrite={false} />
      </points>
      <lineSegments>
        <bufferGeometry><bufferAttribute attach="attributes-position" args={[linePositions, 3]} /></bufferGeometry>
        <lineBasicMaterial color={color} transparent opacity={isActive ? 0.45 : 0.15} blending={THREE.AdditiveBlending} />
      </lineSegments>
      {isActive && agent.dynamicLabel && (
        <Text position={[0, 2.2, 0]} fontSize={0.28} color="#ffffff" anchorX="center" anchorY="middle">
          {agent.dynamicLabel}
        </Text>
      )}
    </group>
  );
}

  function LaserDataStream({ sourceIndex, targetIndex, isFiring }: { sourceIndex: number; targetIndex: number; isFiring: boolean }) {
    const pointsRef = useRef<THREE.Points>(null);
    const count = 350;
    const source = useMemo(() => new THREE.Vector3(...AGENTS[sourceIndex].pos), [sourceIndex]);
    const target = useMemo(() => new THREE.Vector3(...AGENTS[targetIndex].pos), [targetIndex]);
    const sourceColor = AGENTS[sourceIndex].colorActive;

    const curve = useMemo(() => {
      const midpoint = source.clone().lerp(target, 0.5);
      midpoint.y += (Math.random() - 0.5) * 2;
      midpoint.z += (Math.random() - 0.5) * 2;
      return new THREE.QuadraticBezierCurve3(source, midpoint, target);
    }, [source, target]);
    const [positions, offsets, speeds] = useMemo(() => {
      const positions = new Float32Array(count * 3);
      const offsets = new Float32Array(count);
      const speeds = new Float32Array(count);
      for (let index = 0; index < count; index++) {
        offsets[index] = Math.random();
        speeds[index] = 0.4 + Math.random() * 0.8;
        positions.set(curve.getPoint(offsets[index]).toArray(), index * 3);
      }
      return [positions, offsets, speeds];
    }, [curve]);

    useFrame((_, delta) => {
      if (!pointsRef.current || !isFiring) return;
      const array = pointsRef.current.geometry.attributes.position.array as Float32Array;
      for (let index = 0; index < count; index++) {
        offsets[index] = (offsets[index] + delta * speeds[index] * 1.5) % 1;
        const point = curve.getPoint(offsets[index]);
        array.set([point.x, point.y, point.z], index * 3);
      }
      pointsRef.current.geometry.attributes.position.needsUpdate = true;
    });

    if (!isFiring) return null;
    return <points ref={pointsRef}><bufferGeometry><bufferAttribute attach="attributes-position" args={[positions, 3]} /></bufferGeometry><pointsMaterial size={0.065} color={sourceColor} transparent opacity={0.9} blending={THREE.AdditiveBlending} depthWrite={false} /></points>;
  }

function NebulaCore({ state }: { state: CoreState }) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  const stateColors: Record<CoreState, [number, number, number]> = {
    idle: [0.1, 0.2, 0.5],
    listening: [0.2, 0.8, 0.6],
    searching: [0.8, 0.8, 0.2],
    reasoning: [0.4, 0.2, 0.8],
    tool_use: [0.2, 0.5, 0.9],
    error: [0.8, 0.1, 0.1],
    complete: [0.9, 0.9, 0.9],
  };

  const color = useMemo(() => new THREE.Color(...stateColors[state]), [state]);

  useFrame((state_frame) => {
    if (!meshRef.current) return;
    const material = meshRef.current.material as THREE.ShaderMaterial;
    material.uniforms.uTime.value = state_frame.clock.elapsedTime;
    material.uniforms.uColor.value.lerp(color, 0.05);
    material.uniforms.uIntensity.value = state === "idle" ? 0.6 : 1.2;
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[2.5, 64, 64]} />
      <shaderMaterial
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        uniforms={{
          uTime: { value: 0 },
          uColor: { value: new THREE.Color(0, 0, 0) },
          uIntensity: { value: 0.6 },
        }}
        vertexShader={NEBULA_VERTEX_SHADER}
        fragmentShader={NEBULA_FRAGMENT_SHADER}
      />
    </mesh>
  );
}

function ConstellationGrid() {
  const lines = useMemo(() => {
    const coordinates: number[] = [];
    for (let first = 0; first < AGENTS.length; first++) for (let second = first + 1; second < AGENTS.length; second++) coordinates.push(...AGENTS[first].pos, ...AGENTS[second].pos);
    return new Float32Array(coordinates);
  }, []);
  return <lineSegments><bufferGeometry><bufferAttribute attach="attributes-position" args={[lines, 3]} /></bufferGeometry><lineBasicMaterial color="#1a2e47" transparent opacity={0.35} /></lineSegments>;
}

function SmokeField({ quality }: { quality: Quality }) {
  const pointsRef = useRef<THREE.Points>(null);
  const count = quality === "ultra" ? 2200 : quality === "medium" ? 1000 : 300;
  const [positions, basePositions, phases] = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const basePositions = new Float32Array(count * 3);
    const phases = new Float32Array(count);
    for (let index = 0; index < count; index++) {
      const angle = Math.random() * Math.PI * 2;
      const radius = 2.4 + Math.random() * 10.5;
      const offset = index * 3;
      const cloudBand = Math.sin(angle * 3.0 + Math.random() * 2.0) * 1.6;
      const point = [Math.cos(angle) * radius, (Math.random() - 0.5) * 7.5 + cloudBand, Math.sin(angle) * radius * 0.7];
      positions.set(point, offset);
      basePositions.set(point, offset);
      phases[index] = Math.random() * Math.PI * 2;
    }
    return [positions, basePositions, phases];
  }, [count]);

  useFrame((state, delta) => {
    if (!pointsRef.current) return;
    const array = pointsRef.current.geometry.attributes.position.array as Float32Array;
    const time = state.clock.elapsedTime;
    const material = pointsRef.current.material as THREE.ShaderMaterial;
    if (material.uniforms?.uTime) material.uniforms.uTime.value = time;
    for (let index = 0; index < count; index++) {
      const offset = index * 3;
      const phase = phases[index];
      const drift = time * 0.16 + phase;
      const wind = noise3D(basePositions[offset] * 0.09 + time * 0.06, basePositions[offset + 1] * 0.11 + drift * 0.15, basePositions[offset + 2] * 0.09);
      const curl = noise3D(basePositions[offset + 2] * 0.16 + drift, basePositions[offset] * 0.08, time * 0.045);
      const orbit = time * 0.045 + phase * 0.12;
      array[offset] = basePositions[offset] + Math.sin(drift * 1.25) * 1.15 + wind * 1.7 + Math.cos(orbit) * 0.55;
      array[offset + 1] = basePositions[offset + 1] + Math.sin(drift * 0.7) * 0.8 + curl * 1.15 + delta * 0.02;
      array[offset + 2] = basePositions[offset + 2] + Math.cos(drift * 1.05) * 1.0 + wind * 1.25 + Math.sin(orbit) * 0.55;
    }
    pointsRef.current.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <points ref={pointsRef} renderOrder={1}>
      <bufferGeometry><bufferAttribute attach="attributes-position" args={[positions, 3]} /></bufferGeometry>
      <shaderMaterial
        transparent
        depthWrite={false}
        depthTest={false}
        blending={THREE.AdditiveBlending}
        uniforms={{ uTime: { value: 0 } }}
        vertexShader={`
          uniform float uTime;
          varying float vBand;
          void main() {
            vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
            vBand = clamp(position.y * 0.07 + 0.5 + sin(uTime * 0.32 + position.x * 0.35) * 0.26, 0.0, 1.0);
            gl_PointSize = (42.0 / max(1.0, -viewPosition.z)) * (1.5 + vBand * 1.2);
            gl_Position = projectionMatrix * viewPosition;
          }
        `}
        fragmentShader={`
          varying float vBand;
          void main() {
            vec2 centered = gl_PointCoord - 0.5;
            float disc = 1.0 - smoothstep(0.18, 0.5, length(centered));
            vec3 blue = vec3(0.08, 0.24, 1.0);
            vec3 violet = vec3(0.48, 0.08, 1.0);
            vec3 pink = vec3(1.0, 0.18, 0.72);
            vec3 color = mix(violet, blue, smoothstep(0.2, 0.8, vBand));
            color = mix(color, pink, pow(vBand, 5.0) * 0.55);
            gl_FragColor = vec4(color, disc * 0.3);
          }
        `}
      />
    </points>
  );
}

function SingularitySystem() {
  const diskRef = useRef<THREE.Points>(null);
  const wavesRef = useRef<THREE.Group>(null);
  const count = 2200;
  const positions = useMemo(() => {
    const data = new Float32Array(count * 3);
    for (let index = 0; index < count; index++) {
      const radius = 1.2 + Math.pow(Math.random(), 0.65) * 5.8;
      const angle = Math.random() * Math.PI * 2 + radius * 0.72;
      const offset = index * 3;
      data[offset] = Math.cos(angle) * radius;
      data[offset + 1] = (Math.random() - 0.5) * (0.12 + radius * 0.055);
      data[offset + 2] = Math.sin(angle) * radius * 0.3;
    }
    return data;
  }, []);
  const fieldLines = useMemo(() => {
    const data: number[] = [];
    const loops = 18;
    const segments = 48;
    for (let loop = 0; loop < loops; loop++) {
      const phase = (loop / loops) * Math.PI * 2;
      for (let step = 0; step < segments; step++) {
        const a = (step / (segments - 1)) * Math.PI * 2;
        const radius = 1.2 + Math.sin(a * 2 + phase) * 0.25 + loop * 0.13;
        const x = Math.cos(a + phase) * radius;
        const y = Math.sin(a * 1.55 + phase) * (1.0 + loop * 0.045);
        const z = Math.sin(a + phase) * radius * 0.42;
        data.push(x, y, z);
      }
    }
    return new Float32Array(data);
  }, []);

  useFrame((state, delta) => {
    if (!diskRef.current) return;
    const material = diskRef.current.material as THREE.ShaderMaterial;
    if (material.uniforms?.uTime) material.uniforms.uTime.value = state.clock.elapsedTime;
    diskRef.current.rotation.y += delta * 0.16;
    diskRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.18) * 0.045;
    if (wavesRef.current) {
      wavesRef.current.rotation.y -= delta * 0.08;
      wavesRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.22) * 0.12;
      wavesRef.current.rotation.z = Math.cos(state.clock.elapsedTime * 0.16) * 0.08;
    }
  });

  return (
    <group renderOrder={0}>
      <points ref={diskRef}>
        <bufferGeometry><bufferAttribute attach="attributes-position" args={[positions, 3]} /></bufferGeometry>
        <shaderMaterial
          transparent
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          uniforms={{ uTime: { value: 0 } }}
          vertexShader={`
            uniform float uTime;
            varying float vRadius;
            void main() {
              vec3 transformed = position;
              float radius = length(position.xz);
              float angle = atan(position.z, position.x);
              float swirl = uTime * (0.18 + 0.4 / max(radius, 1.0));
              transformed.x = cos(angle + swirl) * radius;
              transformed.z = sin(angle + swirl) * radius * 0.3;
              vRadius = radius;
              vec4 viewPosition = modelViewMatrix * vec4(transformed, 1.0);
              gl_PointSize = (34.0 / max(1.0, -viewPosition.z)) * (1.15 + (1.0 - smoothstep(1.0, 6.5, radius)) * 1.5);
              gl_Position = projectionMatrix * viewPosition;
            }
          `}
          fragmentShader={`
            varying float vRadius;
            void main() {
              float edge = length(gl_PointCoord - 0.5);
              float alpha = 1.0 - smoothstep(0.15, 0.5, edge);
              vec3 inner = vec3(1.0, 0.82, 0.98);
              vec3 outer = vec3(0.34, 0.04, 0.8);
              vec3 color = mix(inner, outer, smoothstep(1.0, 6.5, vRadius));
              gl_FragColor = vec4(color, alpha * 0.58);
            }
          `}
        />
      </points>
      <mesh>
        <sphereGeometry args={[0.48, 32, 32]} />
        <meshBasicMaterial color="#ffffff" toneMapped={false} />
      </mesh>
      <lineSegments>
        <bufferGeometry><bufferAttribute attach="attributes-position" args={[fieldLines, 3]} /></bufferGeometry>
        <lineBasicMaterial color="#c56aff" transparent opacity={0.52} blending={THREE.AdditiveBlending} depthWrite={false} />
      </lineSegments>
      <group ref={wavesRef}>
        {Array.from({ length: 13 }, (_, index) => (
          <mesh key={index} rotation={[index * 0.19, index * 0.47, index * 0.13]} scale={[1, 0.72 + index * 0.035, 1]}>
            <torusGeometry args={[1.35 + index * 0.18, 0.009, 6, 96]} />
            <meshBasicMaterial color={index % 3 === 0 ? "#ff63d8" : "#8e4dff"} transparent opacity={0.34 - index * 0.012} blending={THREE.AdditiveBlending} depthWrite={false} />
          </mesh>
        ))}
      </group>
    </group>
  );
}

function PostProcessing({ isExecuting, quality }: { isExecuting: boolean; quality: Quality }) {
  const bloomRef = useRef<any>(null);
  const aberrationRef = useRef<any>(null);
  const gradeRef = useRef<any>(null);

  useFrame((state) => {
    const intensity = isExecuting ? 2.35 : 1.45;
    if (bloomRef.current) bloomRef.current.intensity = THREE.MathUtils.damp(bloomRef.current.intensity, intensity, 5, state.clock.getDelta());
    if (aberrationRef.current) aberrationRef.current.offset.set(isExecuting ? 0.003 : 0.001, isExecuting ? 0.003 : 0.001);
    if (gradeRef.current) {
      gradeRef.current.exposure = isExecuting ? 1.08 : 1;
      gradeRef.current.contrast = isExecuting ? 0.12 : 0.06;
      gradeRef.current.saturation = isExecuting ? 0.16 : 0.08;
    }
  });

  return <EffectComposer multisampling={quality === "ultra" ? 0 : 0} frameBufferType={THREE.HalfFloatType}>
      {quality !== "low" && <Bloom ref={bloomRef} intensity={1.45} luminanceThreshold={0.18} luminanceSmoothing={0.85} mipmapBlur radius={0.72} levels={7} />}
      {quality !== "low" && <ChromaticAberration ref={aberrationRef} offset={[0.001, 0.001]} radialModulation modulationOffset={0.3} />}
      <ColorGrade ref={gradeRef} exposure={1} contrast={0.06} saturation={0.08} hue={0} />
      <ToneMapping mode={ToneMappingMode.ACES_FILMIC} />
      <Vignette eskil={false} offset={0.25} darkness={0.9} blendFunction={BlendFunction.NORMAL} />
      {quality !== "low" && <Noise opacity={0.03} blendFunction={BlendFunction.SOFT_LIGHT} />}
      <SMAA />
    </EffectComposer>;
}

function FitCamera() {
  const { camera, size } = useThree();

  useEffect(() => {
    const verticalHalfSize = 6.4;
    const horizontalHalfSize = 8.1;
    const verticalFov = THREE.MathUtils.degToRad((camera as THREE.PerspectiveCamera).fov);
    const fitDistance = Math.max(
      verticalHalfSize / Math.tan(verticalFov / 2),
      horizontalHalfSize / (Math.tan(verticalFov / 2) * size.width / size.height),
    ) * 1.12;

    camera.position.set(0, 0, fitDistance);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [camera, size.height, size.width]);

  return null;
}

function AmbientMotion({ isIdle, children }: { isIdle: boolean; children: ReactNode }) {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state, delta) => {
    if (!groupRef.current) return;
    const targetSpeed = isIdle ? 0.049 : 0;
    groupRef.current.rotation.y += delta * targetSpeed;
    groupRef.current.rotation.x = THREE.MathUtils.damp(groupRef.current.rotation.x, isIdle ? 0.12 : 0, 2.5, delta);
    groupRef.current.position.y = Math.sin(state.clock.elapsedTime * (isIdle ? 0.18 : 0)) * (isIdle ? 0.28 : 0);
  });

  return <group ref={groupRef}>{children}</group>;
}

function AmbientCameraTour({ isIdle }: { isIdle: boolean }) {
  const { camera, size } = useThree();
  const tourTime = useRef(0);
  const previousIdle = useRef(false);

  useFrame((state, delta) => {
    if (!isIdle) {
      previousIdle.current = false;
      return;
    }

    if (!previousIdle.current) {
      tourTime.current = 0;
      previousIdle.current = true;
    }

    const perspectiveCamera = camera as THREE.PerspectiveCamera;
    const verticalHalfSize = 6.4;
    const horizontalHalfSize = 8.1;
    const verticalFov = THREE.MathUtils.degToRad(perspectiveCamera.fov);
    const fitDistance = Math.max(
      verticalHalfSize / Math.tan(verticalFov / 2),
      horizontalHalfSize / (Math.tan(verticalFov / 2) * size.width / size.height),
    ) * 1.12;
    const pace = (0.055 + (Math.sin(state.clock.elapsedTime * 0.16) * 0.5 + 0.5) * 0.035) * 1.4;
    tourTime.current += delta * pace;

    const time = tourTime.current;
    const horizontalDrift = Math.sin(time * 0.72) * 2.2 + Math.sin(time * 1.4) * 0.55;
    const verticalDrift = Math.sin(time * 0.93 + 1.1) * 0.75;
    const zoom = Math.sin(time * 0.56) * 2.25 + Math.sin(time * 1.1) * 0.55;
    const targetX = Math.sin(time * 0.62 + 0.8) * 0.8;
    const targetY = Math.sin(time * 0.78) * 0.35;

    camera.position.x = THREE.MathUtils.damp(camera.position.x, horizontalDrift, 1.3, delta);
    camera.position.y = THREE.MathUtils.damp(camera.position.y, verticalDrift, 1.3, delta);
    camera.position.z = THREE.MathUtils.damp(camera.position.z, fitDistance + zoom, 1.3, delta);
    camera.lookAt(targetX, targetY, 0);
  });

  return null;
}

function Scene({ activeIndex, targetIndex, isExecuting, isIdle, coreState, quality }: { activeIndex: number; targetIndex: number; isExecuting: boolean; isIdle: boolean; coreState: CoreState; quality: Quality }) {

  return <Canvas camera={{ position: [0, 0, 14], fov: 45}}>
    <FitCamera />
    <AmbientCameraTour isIdle={isIdle} />
    <color attach="background" args={["#010206"]} />
    <ambientLight intensity={0.2} />
    <Environment preset="city" />
    <Stars radius={80} depth={50} count={5000} factor={3} fade speed={isIdle ? 0.168 : 0.5} />
    <AmbientMotion isIdle={isIdle}>
      <SmokeField quality={quality} />
      <SingularitySystem />
      <NebulaCore state={coreState} />
      <ConstellationGrid />
      {AGENTS.map((agent, index) => (
        <Float key={agent.id} speed={isIdle ? 0.49 : 1.5} floatIntensity={isIdle ? 0.5 : 0.25}>
          <AgentFixture agent={agent} isActive={!isIdle && isExecuting && index === activeIndex} />
          <AgentCloud agent={agent} isActive={!isIdle && isExecuting && index === activeIndex} isTarget={!isIdle && isExecuting && index === targetIndex} quality={quality} />
        </Float>
      ))}
    </AmbientMotion>
    <LaserDataStream sourceIndex={activeIndex} targetIndex={targetIndex} isFiring={isExecuting} />
    <PostProcessing isExecuting={isExecuting} quality={quality} />
    <OrbitControls enabled={!isIdle} enablePan={false} maxDistance={22} minDistance={6} />
  </Canvas>;
}

export default function App() {
  const [activeIndex, setActiveIndex] = useState(0);
  const [targetIndex, setTargetIndex] = useState(1);
  const [isExecuting, setIsExecuting] = useState(false);
  const [isHudVisible, setIsHudVisible] = useState(true);
  const [isIdle, setIsIdle] = useState(false);
  const [quality, setQuality] = useState<Quality>("ultra");
  const [throughput, setThroughput] = useState(312);
  const [logs, setLogs] = useState(["09:41  init    grid-swarm-01 online", "09:41  grant   syncpool 10.20M actions ok", "09:42  agent   calendar_sync standing by"]);
  const [copilotOpen, setCopilotOpen] = useState(false);
  const [copilotInput, setCopilotInput] = useState("");
  const [aiProvider, setAiProvider] = useState<AiProvider>("ollama");
  const [aiModel, setAiModel] = useState("llama3.2");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [aiBaseUrl, setAiBaseUrl] = useState("http://127.0.0.1:11434");
  const [anthropicKey, setAnthropicKey] = useState("");
  const [aiBusy, setAiBusy] = useState(false);
  const [coreState, setCoreState] = useState<CoreState>("idle");
  const [copilotMessages, setCopilotMessages] = useState<CopilotMessage[]>([
    { role: "ai", text: "Copilot online. Ask for status, focus an agent, or run a workflow." },
  ]);

  useEffect(() => {
    const unlisten = listen<StreamPayload>("ai-chunk", (event) => {
      const { text, state, is_final } = event.payload;
      
      setCopilotMessages((current) => {
        const last = current[current.length - 1];
        if (last && last.role === "ai") {
          return [...current.slice(0, -1), { role: "ai", text: last.text + text }];
        }
        return [...current, { role: "ai", text: text }];
      });
      
      if (state) setCoreState(state as CoreState);
      if (is_final) {
        setAiBusy(false);
        setCoreState("complete");
      }
    });
    return () => { unlisten.then((fn) => fn()); };
  }, []);

  function triggerExecution(from: number, to: number, taskName: string) {
    setActiveIndex(from); setTargetIndex(to); setIsExecuting(true);
    setLogs((current) => [`${new Date().toLocaleTimeString()}  exec    ${AGENTS[from].name} -> ${AGENTS[to].name} [${taskName}]`, ...current].slice(0, 15));
    setThroughput((current) => current + Math.floor(Math.random() * 80 + 20));
    window.setTimeout(() => setIsExecuting(false), 2400);
  }

  async function runCopilotCommand(command: string) {
    const normalized = command.trim().toLowerCase();
    if (!normalized) return;

    setAiBusy(true);
    setCoreState("listening");

    let response = "I can check status, focus an agent, run a workflow, or ask the connected AI provider.";
    const requestedAgent = AGENTS.find((agent) => normalized.includes(agent.id.toLowerCase()) || normalized.includes(agent.name.split(" ")[0]));

    if (normalized.includes("status") || normalized.includes("health")) {
      setCoreState("searching");
      response = `System is online at ${throughput} ops/sec. ${isExecuting ? "Execution stream is active." : "All agents are standing by."}`;
      setTimeout(() => { setCoreState("idle"); setAiBusy(false); }, 2000);
    } else if (normalized.includes("focus") && requestedAgent) {
      setCoreState("reasoning");
      setTargetIndex(AGENTS.indexOf(requestedAgent));
      response = `Tracking ${requestedAgent.name}. The camera target is queued for the next execution.`;
      setTimeout(() => { setCoreState("idle"); setAiBusy(false); }, 1500);
    } else if (normalized.includes("run") || normalized.includes("start") || normalized.includes("execute")) {
      setCoreState("tool_use");
      const target = requestedAgent ?? AGENTS[(activeIndex + 1) % AGENTS.length];
      triggerExecution(activeIndex, AGENTS.indexOf(target), requestedAgent ? `${target.id.toLowerCase()}_analysis` : "synthesis_report");
      response = `Execution started: ${AGENTS[activeIndex].name} -> ${target.name}.`;
      setTimeout(() => { setCoreState("idle"); setAiBusy(false); }, 2400);
    } else if (normalized.includes("tour") || normalized.includes("ambient")) {
      response = "Ambient tour is automatic after inactivity. Move the pointer to return to the live controls.";
      setAiBusy(false);
    } else {
      setCopilotInput("");
      if (aiProvider === "ollama") {
        setCoreState("reasoning");
        try {
          await invoke("stream_ai", {
            request: {
              provider: aiProvider,
              prompt: command,
              model: aiModel,
              apiKey: null,
              baseUrl: aiBaseUrl,
            },
          });
        } catch (e) {
          setCoreState("error");
          setCopilotMessages((current) => [...current, { role: "ai", text: `Error: ${e}` }]);
          setAiBusy(false);
        }
        return;
      } else {
        setCoreState("reasoning");
        try {
          let result;
          if (aiProvider === "perplexity") {
            result = await invoke<{ provider: string; model: string; text: string }>("ask_perplexity", {
              request: {
                provider: aiProvider,
                prompt: command,
                model: aiModel,
                apiKey: null,
                baseUrl: null,
              },
            });
          } else if (aiProvider === "perplexity_cloud") {
            result = await invoke<{ provider: string; model: string; text: string }>("ask_perplexity_cloud", {
              request: {
                provider: aiProvider,
                prompt: command,
                model: aiModel,
                apiKey: null,
                baseUrl: null,
              },
            });
          } else {
            result = await invoke<{ provider: string; model: string; text: string }>("ask_ai", {
              request: {
                provider: aiProvider,
                prompt: command,
                model: aiModel,
                apiKey: aiProvider === "anthropic" ? anthropicKey : null,
                baseUrl: null,
              },
            });
          }

          const aiText = result.text;
          const actions = aiText.matchAll(/\[ACTION: (\w+), (?:FROM: (\w+), TO: (\w+)|TARGET: (\w+)|VALUE: (\d+))\]/g);
          for (const match of actions) {
            const [_, type, from, to, target, value] = match;
            if (type === "EXECUTE") {
              const fromIdx = AGENTS.findIndex(a => a.id === from?.toUpperCase());
              const toIdx = AGENTS.findIndex(a => a.id === to?.toUpperCase());
              if (fromIdx !== -1 && toIdx !== -1) {
                AGENTS[fromIdx].dynamicLabel = "Sourcing...";
                AGENTS[toIdx].dynamicLabel = "Processing...";
                triggerExecution(fromIdx, toIdx, "ai_orchestrated_stream");
                window.setTimeout(() => {
                  AGENTS[fromIdx].dynamicLabel = undefined;
                  AGENTS[toIdx].dynamicLabel = undefined;
                }, 2400);
              }
            } else if (type === "FOCUS") {
              const targetIdx = AGENTS.findIndex(a => a.id === target?.toUpperCase());
              if (targetIdx !== -1) {
                AGENTS[targetIdx].dynamicLabel = "Focused";
                setTargetIndex(targetIdx);
                window.setTimeout(() => {
                  AGENTS[targetIdx].dynamicLabel = undefined;
                }, 3000);
              }
            } else if (type === "CLEAR_LOGS") {
              setLogs([]);
            } else if (type === "SET_THROUGHPUT") {
              if (value) setThroughput(parseInt(value, 10));
            }
          }

          setCopilotMessages((current) => [...current, { role: "ai", text: `[${result.provider} / ${result.model}] ${aiText}` }]);
          setCoreState("complete");
        } catch (error) {
          setCoreState("error");
          setCopilotMessages((current) => [...current, { role: "ai", text: String(error) }]);
        } finally {
          setAiBusy(false);
        }
        return;
      }
    }

    setCopilotMessages((current) => [...current, { role: "user", text: command }, { role: "ai", text: response }]);
    setAiBusy(false);
  }

  useEffect(() => {
    async function fetchModels() {
      if (aiProvider !== "ollama") return;
      try {
        const response = await fetch(`${aiBaseUrl}/api/tags`);
        const data = await response.json();
        const models = data.models.map((m: any) => m.name);
        setAvailableModels(models);
        if (models.length > 0 && !models.includes(aiModel)) {
          setAiModel(models[0]);
        }
      } catch (e) {
        console.error("Failed to fetch Ollama models", e);
      }
    }
    fetchModels();
  }, [aiProvider, aiBaseUrl]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      const from = Math.floor(Math.random() * AGENTS.length);
      let to = Math.floor(Math.random() * AGENTS.length);
      while (to === from) to = Math.floor(Math.random() * AGENTS.length);
      triggerExecution(from, to, "token_stream");
    }, 3800);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    let hideTimer = 0;
    const revealHud = () => {
      setIsHudVisible(true);
      setIsIdle(false);
      window.clearTimeout(hideTimer);
      hideTimer = window.setTimeout(() => {
        setIsHudVisible(false);
        setIsIdle(true);
      }, 5000);
    };

    window.addEventListener("pointermove", revealHud);
    window.addEventListener("keydown", revealHud);
    revealHud();
    return () => {
      window.removeEventListener("pointermove", revealHud);
      window.removeEventListener("keydown", revealHud);
      window.clearTimeout(hideTimer);
    };
  }, []);

  return (
    <main className={`hud-container${isHudVisible ? "" : " hud-faded"}`}>
      <div className="canvas-wrapper">
        <Scene activeIndex={activeIndex} targetIndex={targetIndex} isExecuting={isExecuting} isIdle={isIdle} coreState={coreState} quality={quality} />
      </div>
      <header className="hud-top-bar">
        <div className="hud-brand">
          <span className="hud-status-bulb" />
          <b>grid-swarm-01</b>
          <small>actions 12,320</small>
          <small>profile 0 beta</small>
        </div>
        <div className="hud-stats">
          <span>pending <b>3.47</b></span>
          <span>wait <b>0020</b></span>
          <span>cluster <b>ONLINE</b></span>
        </div>
      </header>
      <footer className="hud-bottom-deck">
        <section className="hud-log-panel">
          <div className="panel-title">NODE LOG STREAM</div>
          <div className="log-scroll">
            {logs.map((log, index) => (
              <div key={`${log}-${index}`} className="log-line">{log}</div>
            ))}
          </div>
        </section>
        <section className="hud-metrics-panel">
          <div className="panel-title">SYSTEM THROUGHPUT</div>
          <div className="metric-number">{throughput} <small>ops/sec</small></div>
          <div className="metric-bar">
            <div className="metric-fill" style={{ width: `${Math.min(throughput / 8, 100)}%` }} />
          </div>
          <div className="metric-row">
            <span>PEAK 96%</span>
            <span>DRAIN 0.02ms</span>
          </div>
        </section>
        <section className="hud-action-controls">
          <button onClick={() => triggerExecution(0, 1, "inbox_triage")}>TRIGGER INBOX TRIAGE</button>
          <button onClick={() => triggerExecution(2, 3, "report_build")}>BUILD SYNTHESIS REPORT</button>
          <button className="copilot-toggle" onClick={() => setCopilotOpen((open) => !open)}>
            AI COPILOT {copilotOpen ? "-" : "+"}
          </button>
        </section>
      </footer>
      {copilotOpen && (
        <section className="copilot-panel">
          <div className="panel-title">AI OPERATIONS COPILOT</div>
          <div className="copilot-provider-row">
            <select aria-label="AI provider" value={aiProvider} onChange={(event) => setAiProvider(event.target.value as AiProvider)}>
              <option value="ollama">OLLAMA / LOCAL</option>
              <option value="anthropic">ANTHROPIC / CLOUD</option>
              <option value="perplexity">PERPLEXITY / WEB</option>
              <option value="perplexity_cloud">PERPLEXITY / CLOUD</option>
            </select>
            <select
              aria-label="AI model"
              value={aiModel}
              onChange={(event) => setAiModel(event.target.value)}
              style={{ minWidth: '120px' }}
            >
              {aiProvider === "perplexity_cloud" ? (
                Object.entries(CLOUD_MODELS).map(([name, id]) => (
                  <option key={id} value={id}>{name}</option>
                ))
              ) : availableModels.length > 0 ? (
                availableModels.map(model => <option key={model} value={model}>{model}</option>)
              ) : (
                <option value={aiModel}>{aiModel} (none found)</option>
              )}
            </select>
          </div>
          <div className="copilot-settings-row">
            {aiProvider === "ollama" ? (
              <input className="copilot-setting" aria-label="Ollama base URL" value={aiBaseUrl} onChange={(event) => setAiBaseUrl(event.target.value)} placeholder="Ollama URL" />
            ) : (
              <input className="copilot-setting" aria-label="Anthropic API key" type="password" value={anthropicKey} onChange={(event) => setAnthropicKey(event.target.value)} placeholder="Anthropic API key" />
            )}
            <select aria-label="GPU Quality" value={quality} onChange={(event) => setQuality(event.target.value as Quality)} className="copilot-setting">
              <option value="low">LOW QUALITY</option>
              <option value="medium">MEDIUM QUALITY</option>
              <option value="ultra">ULTRA QUALITY</option>
            </select>
          </div>
          <div className="copilot-messages">
            {copilotMessages.map((message, index) => (
              <div key={`${message.role}-${index}`} className={`copilot-message ${message.role}`}>
                <span>{message.role === "ai" ? "AI" : "YOU"}</span>
                {message.text}
              </div>
            ))}
          </div>
          <form className="copilot-form" onSubmit={(event) => { event.preventDefault(); runCopilotCommand(copilotInput); }}>
            <input aria-label="Ask the operations copilot" value={copilotInput} onChange={(event) => setCopilotInput(event.target.value)} placeholder="ask: status / focus inbox / summarize" />
            <button type="submit" aria-label="Send copilot command" disabled={aiBusy}>{aiBusy ? "..." : "SEND"}</button>
          </form>
        </section>
      )}
    </main>
  );
}
