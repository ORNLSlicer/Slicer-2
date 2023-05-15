#version 440
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 uv;
out vec4 vColor;
out vec3 vNormal;
out vec3 fragPos;
out vec2 texcoord_uv;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

mat3 rotation;

void main()
{
    vColor = color;
    rotation = mat3(model[0][0], model[0][1], model[0][2],
                    model[1][0], model[1][1], model[1][2],
                    model[2][0], model[2][1], model[2][2]);
    vNormal = normalize(rotation * normal);
    fragPos = vec3(model * vec4(position, 1.0));
    gl_Position = projection * view * model * vec4(position, 1.0);
    texcoord_uv = uv;
}
