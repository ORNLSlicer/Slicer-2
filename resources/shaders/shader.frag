#version 440

in vec4 vColor;
in vec3 fragPos;
in vec3 vNormal;
in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec2 texcoord_uv;

in vec3 bary;
out vec4 fColor;

uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform sampler2D textureSamp;
uniform float ambientStrength;
uniform bool usingSolidWireframeMode;

void main()
{

        // Ambient
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 norm = normalize(vNormal);
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(vNormal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;


    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 64);
    vec3 specular = specularStrength * spec * lightColor;


    float nearD = min(min(bary[0],bary[1]),bary[2]);

    //Dictates how bold/wide the edges of the wireframe are, the more negative
    //the less bold
    float edgeIntensityCoefficient = -25;

    //If we are not using solid wireframe mode, this equation equals 0
    //and no edges are rendered.
    float edgeIntensity =  exp2(edgeIntensityCoefficient*nearD) * float(usingSolidWireframeMode);

    vec4 color = vec4(1,0,0,1);

    vec3 result = (ambient + diffuse + specular) * vec3(vColor) * (1.0-edgeIntensity);
    fColor =  vec4(0.1,0.1,0.1,0) + vec4(result, vColor.a);


}
