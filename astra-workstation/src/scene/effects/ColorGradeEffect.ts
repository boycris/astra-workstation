import { BlendFunction, Effect } from "postprocessing";
import { Color, Uniform } from "three";

const fragmentShader = `
uniform float uExposure;
uniform float uContrast;
uniform float uSaturation;
uniform float uHue;
uniform vec3 uShadowTint;
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
vec3 hueRotate(vec3 color, float angle) {
  const vec3 axis = vec3(0.57735);
  float cosine = cos(angle);
  float sine = sin(angle);
  return color * cosine + cross(axis, color) * sine + axis * dot(axis, color) * (1.0 - cosine);
}
void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  vec3 color = inputColor.rgb * uExposure;
  color = (color - 0.5) * (1.0 + uContrast) + 0.5;
  float luminance = dot(color, LUMA);
  color = mix(vec3(luminance), color, 1.0 + uSaturation);
  color = hueRotate(color, uHue);
  float shadowMask = 1.0 - smoothstep(0.0, 0.35, luminance);
  color += uShadowTint * shadowMask * 0.12;
  outputColor = vec4(max(color, 0.0), inputColor.a);
}`;

export class ColorGradeEffect extends Effect {
  constructor(exposure = 1, contrast = 0.06, saturation = 0.08, hue = 0) {
    super("ColorGradeEffect", fragmentShader, {
      blendFunction: BlendFunction.NORMAL,
      uniforms: new Map<string, Uniform<any>>([
        ["uExposure", new Uniform(exposure)],
        ["uContrast", new Uniform(contrast)],
        ["uSaturation", new Uniform(saturation)],
        ["uHue", new Uniform(hue)],
        ["uShadowTint", new Uniform(new Color("#1a1040"))],
      ]),
    });
  }

  set exposure(value: number) { this.uniforms.get("uExposure")!.value = value; }
  set contrast(value: number) { this.uniforms.get("uContrast")!.value = value; }
  set saturation(value: number) { this.uniforms.get("uSaturation")!.value = value; }
  set hue(value: number) { this.uniforms.get("uHue")!.value = value; }
}
