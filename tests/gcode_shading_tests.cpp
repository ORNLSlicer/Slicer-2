#include <QDir>
#include <QGuiApplication>
#include <QImage>
#include <QMatrix4x4>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QOpenGLFunctions_4_4_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLVersionFunctionsFactory>
#include <QSurfaceFormat>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "graphics/support/shape_factory.h"
#include "test_utils.h"

namespace {
QImage renderBead(QOpenGLFunctions_4_4_Core& gl, QOpenGLShaderProgram& program, const QColor& color,
                  bool preserve_hue) {
    std::vector<float> vertices, colors, normals;
    ORNL::ShapeFactory::appendLinearBead(0.8f, 2.0f, 0.5f, QVector3D(-1, 0, 0), QVector3D(1, 0, 0), color, vertices,
                                         colors, normals);

    QOpenGLFramebufferObject framebuffer(QSize(480, 320), QOpenGLFramebufferObject::CombinedDepthStencil);
    framebuffer.bind();
    gl.glViewport(0, 0, 480, 320);
    gl.glDisable(GL_BLEND);
    gl.glDisable(GL_DITHER);
    gl.glEnable(GL_DEPTH_TEST);
    gl.glClearColor(0.94f, 0.94f, 1.0f, 1.0f);
    gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    GLuint vao;
    std::array<GLuint, 3> buffers;
    gl.glGenVertexArrays(1, &vao);
    gl.glBindVertexArray(vao);
    gl.glGenBuffers(buffers.size(), buffers.data());
    const std::array<const std::vector<float>*, 3> attributes {&vertices, &normals, &colors};
    for (GLuint attribute = 0; attribute < attributes.size(); ++attribute) {
        const auto& data = *attributes[attribute];
        gl.glBindBuffer(GL_ARRAY_BUFFER, buffers[attribute]);
        gl.glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_STATIC_DRAW);
        gl.glEnableVertexAttribArray(attribute);
        gl.glVertexAttribPointer(attribute, attribute == 2 ? 4 : 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    }

    QMatrix4x4 model, view, projection;
    const QVector3D camera(3, -4, 3);
    view.lookAt(camera, QVector3D(), QVector3D(0, 0, 1));
    projection.ortho(-1.8f, 1.8f, -1.2f, 1.2f, 0.1f, 100.0f);
    program.bind();
    program.setUniformValue("model", model);
    program.setUniformValue("view", view);
    program.setUniformValue("projection", projection);
    program.setUniformValue("stackingAxis", QVector3D(0, 0, 1));
    program.setUniformValue("usingOverhangMode", false);
    program.setUniformValue("renderingPartObject", false);
    program.setUniformValue("usingSolidWireframeMode", false);
    program.setUniformValue("lightColor", QVector3D(1, 1, 1));
    program.setUniformValue("lightPos", QVector3D(-3, -3, 5));
    program.setUniformValue("viewPos", camera);
    program.setUniformValue("ambientStrength", 0.4f);
    program.setUniformValue("preserveColorHue", preserve_hue);
    gl.glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);
    const QImage result = framebuffer.toImage();
    program.release();
    gl.glBindVertexArray(0);
    gl.glDeleteBuffers(buffers.size(), buffers.data());
    gl.glDeleteVertexArrays(1, &vao);
    framebuffer.release();
    return result;
}

bool checkShadedColor(const QImage& image, const QImage& original, const QColor& base) {
    const QColor background = image.pixelColor(0, 0);
    const std::array<float, 3> base_rgb {base.redF(), base.greenF(), base.blueF()};
    const float base_peak   = *std::max_element(base_rgb.begin(), base_rgb.end());
    const float base_low    = *std::min_element(base_rgb.begin(), base_rgb.end());
    const float base_chroma = base_peak - base_low;
    float darkest           = 1.0f;
    float brightest         = 0.0f;
    int bead_pixels         = 0;
    int chromatic_pixels    = 0;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const QColor pixel = image.pixelColor(x, y);
            if (pixel == background) { continue; }
            ++bead_pixels;
            const std::array<float, 3> rgb {pixel.redF(), pixel.greenF(), pixel.blueF()};
            const float peak   = *std::max_element(rgb.begin(), rgb.end());
            const float low    = *std::min_element(rgb.begin(), rgb.end());
            const float chroma = peak - low;
            darkest            = std::min(darkest, peak);
            brightest          = std::max(brightest, peak);
            if (chroma > 0.02f) { ++chromatic_pixels; }
            for (int channel = 0; channel < 3; ++channel) {
                if ((base_chroma > 0 && chroma > 0.02f &&
                     std::abs(((rgb[channel] - low) / chroma) - ((base_rgb[channel] - base_low) / base_chroma)) >
                         0.02f) ||
                    (base_chroma == 0 && chroma > (1.0f / 255.0f))) {
                    std::cerr << "Face changed the configured hue for " << base.name().toStdString() << ": rendered "
                              << pixel.name().toStdString() << '\n';
                    return false;
                }
            }
            const QColor original_pixel    = original.pixelColor(x, y);
            const float original_luminance = (0.2126f * original_pixel.redF()) + (0.7152f * original_pixel.greenF()) +
                                             (0.0722f * original_pixel.blueF());
            const float luminance          = (0.2126f * rgb[0]) + (0.7152f * rgb[1]) + (0.0722f * rgb[2]);
            if (std::abs(luminance - original_luminance) > (2.0f / 255.0f)) {
                std::cerr << "Shading changed the original face brightness for " << base.name().toStdString() << '\n';
                return false;
            }
        }
    }
    return ORNL::Testing::expect(bead_pixels > 1000, "The regression must render a visible bead mesh.") &&
           ORNL::Testing::expect(base_chroma == 0 || chromatic_pixels > 1000,
                                 "Colored paths must retain their configured hue on visible faces.") &&
           ORNL::Testing::expect(base_peak == 0 || brightest - darkest > 0.1f,
                                 "Hue-preserving shading must retain light/dark depth cues.");
}
}  // namespace

int main(int argc, char* argv[]) {
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) { qputenv("QT_QPA_PLATFORM", "offscreen"); }
    QGuiApplication app(argc, argv);
    Q_INIT_RESOURCE(shaders);

    QSurfaceFormat format;
    format.setVersion(4, 4);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QOpenGLContext context;
    context.setFormat(format);
    if (!context.create()) {
        std::cerr << "OpenGL 4.4 is unavailable; offscreen shading regression cannot run.\n";
        return 77;
    }
    QOffscreenSurface surface;
    surface.setFormat(context.format());
    surface.create();
    if (!surface.isValid() || !context.makeCurrent(&surface)) {
        std::cerr << "An offscreen OpenGL surface is unavailable.\n";
        return 77;
    }
    auto* gl = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_4_Core>(&context);
    if (gl == nullptr || !gl->initializeOpenGLFunctions()) { return EXIT_FAILURE; }

    QOpenGLShaderProgram program;
    if (!program.addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/vert") ||
        !program.addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/frag") || !program.link()) {
        std::cerr << program.log().toStdString();
        return EXIT_FAILURE;
    }
    if (!ORNL::Testing::expect(program.uniformLocation("preserveColorHue") >= 0,
                               "The production shader must support hue-preserving G-code lighting.")) {
        return EXIT_FAILURE;
    }

    bool passed = true;
    const QColor connection(0, 250, 154);
    const QImage original = renderBead(*gl, program, connection, false);
    for (const QColor& color : {connection, QColor(255, 128, 0), QColor(179, 128, 255), QColor(0, 0, 255),
                                QColor(0, 0, 0), QColor(100, 100, 100), QColor(255, 255, 255)}) {
        const QImage original_color = color == connection ? original : renderBead(*gl, program, color, false);
        const QImage shaded         = renderBead(*gl, program, color, true);
        passed &= checkShadedColor(shaded, original_color, color);
        const QString preview_dir = qEnvironmentVariable("ORNLSLICER_SHADING_PREVIEW_DIR");
        if (color == connection && !preview_dir.isEmpty()) {
            QDir().mkpath(preview_dir);
            original.save(preview_dir + "/gcode-shading-before.png");
            shaded.save(preview_dir + "/gcode-shading-after.png");
        }
    }
    passed &= ORNL::Testing::expect(original == renderBead(*gl, program, connection, false),
                                    "Rendering without hue preservation must retain the original scene shading.");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
