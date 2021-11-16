#version 330

uniform sampler2D image;
in vec2 uv;
out vec4 out_color;

void main() {
    vec4 color = texture(image, uv);
    out_color = vec4(color.xyz, 1.0);
    //out_color = vec4(uv, 0.0, 1.0);
}