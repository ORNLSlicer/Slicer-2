#version 440
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;
layout(location = 3) in vec2 uv;
out vec4 vColor;
out vec3 fragPos;
out vec3 vNormal;
out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec2 texcoord_uv;
out vec3 bary;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 stackingAxis;
uniform float overhangAngle;
uniform bool usingOverhangMode;
uniform bool renderingPartObject;
mat3 rotation;

void main()
{
    vec4 temp = (model * vec4(position, 1));
    vWorldPos = vec3(temp.x, temp.y, temp.z);
    rotation = mat3(model[0][0], model[0][1], model[0][2],
                    model[1][0], model[1][1], model[1][2],
                    model[2][0], model[2][1], model[2][2]);
    vNormal = normalize(normal);
    vColor = color;
    vWorldNormal = rotation * normal;
    fragPos = vec3(model * vec4(position, 1.0));
    gl_Position = projection * view * model * vec4(position, 1.0);
    texcoord_uv = uv;
    //We need the know the baryocentric coordinate of each pixel to determine how
    //close each pixel is to an edge of a triangle
    bary = vec3(gl_VertexID % 3 == 0, gl_VertexID % 3 == 1, gl_VertexID % 3 == 2);

    //Determine if this vertex is facing downwards
    float upward = dot(stackingAxis, vWorldNormal);

    //Z pointing down
    if (upward < 0.0 && usingOverhangMode && renderingPartObject)
    {
        vec4 overhangColor = vec4(255, 0, 0, 255);
        float M_PI = 3.14159265358979323846;
        float faceAngle;
        float val = dot(stackingAxis, vWorldNormal) / (length(vWorldNormal) * length(stackingAxis));
        float clampedVal = clamp(val, -1.0, 1.0);

        if (val <= -1.0)
            faceAngle = M_PI;
        else if (val >= 1.0)
            faceAngle = 0;
        else
            faceAngle = acos(val);
        //Adjust the faceAngle by 90 degrees so that straight down is at 90 degrees
        faceAngle = faceAngle - (M_PI / 2.0);
        //if the angle of the face is greater than roughly 40 degrees (where 0 degrees would mean a face that is perpendicular
        //to the stacking axis)
        if (faceAngle > overhangAngle)
        {
            vColor = overhangColor;
        }

    }


}
