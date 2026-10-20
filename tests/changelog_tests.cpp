#include <QApplication>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QPushButton>
#include <QTemporaryDir>
#include <QTextBlock>
#include <QTextBrowser>
#include <QTextFragment>
#include <cstdlib>

#include "test_utils.h"
#include "windows/changelog.h"

int main(int argc, char* argv[]) {
    qputenv("QT_QPA_PLATFORM", QByteArray("offscreen"));
    QApplication app(argc, argv);
    Q_INIT_RESOURCE(docs);

    // Installed builds must display the bundled history without a source checkout.
    QTemporaryDir directory;
    bool passed = ORNL::Testing::expect(directory.isValid() && QDir::setCurrent(directory.path()),
                                        "The test should run outside the source tree.");
    QFile changelog(":/docs/CHANGELOG.md");
    passed &= ORNL::Testing::expect(changelog.open(QIODevice::ReadOnly), "The changelog should be bundled.");
    passed &= ORNL::Testing::expect(changelog.readAll().contains("## Unreleased"),
                                    "The bundled changelog should include Unreleased.");

    QWidget parent;
    ORNL::ChangelogWindow window(&parent);
    window.show();
    app.processEvents();

    auto* browser = window.findChild<QTextBrowser*>();
    auto* buttons = window.findChild<QDialogButtonBox*>();
    passed &= ORNL::Testing::expect(browser != nullptr && buttons != nullptr,
                                    "The changelog should provide a browser and Close button.");
    if (browser == nullptr || buttons == nullptr) return EXIT_FAILURE;

    passed &= ORNL::Testing::expect(window.isWindow() && !window.isModal() && window.parentWidget() == &parent,
                                    "The changelog should be a nonmodal window owned by its parent.");
    passed &= ORNL::Testing::expect(browser->isReadOnly() && browser->openExternalLinks(),
                                    "The changelog should be read-only with working external links.");
    const QString text = browser->toPlainText();
    passed &= ORNL::Testing::expect(
        text.contains("Unreleased") && text.contains("Help > Changelog") && !text.contains("## Unreleased"),
        "The changelog should render Markdown rather than show its source.");

    bool has_heading = false;
    bool has_link    = false;
    for (QTextBlock block = browser->document()->begin(); block.isValid(); block = block.next()) {
        if (block.text() == "Unreleased") has_heading = block.blockFormat().headingLevel() == 2;
        for (auto fragment = block.begin(); !fragment.atEnd(); ++fragment) {
            if (fragment.fragment().charFormat().anchorHref() == "https://github.com/ORNLSlicer/ORNLSlicer/issues/309")
                has_link = fragment.fragment().charFormat().fontUnderline() &&
                           !fragment.fragment().charFormat().hasProperty(QTextFormat::ForegroundBrush);
        }
    }
    passed &= ORNL::Testing::expect(has_heading && has_link,
                                    "Markdown headings and underlined, theme-colored issue links should be retained.");
    passed &= ORNL::Testing::expect(window.isVisible(), "Opening the changelog should show it.");
    buttons->button(QDialogButtonBox::Close)->click();
    passed &= ORNL::Testing::expect(!window.isVisible(), "Close should hide the changelog.");
    window.showNormal();
    app.processEvents();
    passed &= ORNL::Testing::expect(window.isVisible() && browser->toPlainText() == text,
                                    "Reopening should preserve the bundled changelog.");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
