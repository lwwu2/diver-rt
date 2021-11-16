#version 330

in vec2 in_position;
in vec2 in_uv;
out vec2 uv;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    uv = in_uv;
}