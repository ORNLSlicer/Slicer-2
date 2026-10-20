#include "windows/changelog.h"

#include <QDialogButtonBox>
#include <QFile>
#include <QIcon>
#include <QTextBlock>
#include <QTextBrowser>
#include <QTextCursor>
#include <QTextFragment>
#include <QVBoxLayout>

namespace ORNL {
ChangelogWindow::ChangelogWindow(QWidget* parent) : QDialog(parent, Qt::Window) {
    setWindowTitle("Changelog");
    setWindowIcon(QIcon(":/icons/ornlslicer_logo.png"));
    resize(850, 650);

    auto* browser = new QTextBrowser(this);
    browser->setOpenExternalLinks(true);

    QFile changelog(":/docs/CHANGELOG.md");
    if (changelog.open(QIODevice::ReadOnly | QIODevice::Text)) {
        browser->setMarkdown(QString::fromUtf8(changelog.readAll()));
        // Inherit the theme's text color instead of Markdown's fixed blue link color.
        for (QTextBlock block = browser->document()->begin(); block.isValid(); block = block.next()) {
            for (auto fragment = block.begin(); !fragment.atEnd(); ++fragment) {
                const QTextFragment text = fragment.fragment();
                auto format              = text.charFormat();
                if (!format.isAnchor()) continue;
                format.clearForeground();
                format.setFontUnderline(true);
                QTextCursor cursor(browser->document());
                cursor.setPosition(text.position());
                cursor.setPosition(text.position() + text.length(), QTextCursor::KeepAnchor);
                cursor.setCharFormat(format);
            }
        }
    }
    else { browser->setPlainText("The bundled changelog could not be opened."); }

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    auto* layout = new QVBoxLayout(this);
    layout->addWidget(browser);
    layout->addWidget(buttons);
}
}  // namespace ORNL
