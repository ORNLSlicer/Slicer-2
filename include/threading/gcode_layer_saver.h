#pragma once

#include <QString>
#include <QThread>

#include <qtmetamacros.h>

namespace ORNL {
class GCodeLayerSaver : public QThread {
    Q_OBJECT
   public:
    GCodeLayerSaver(QString path, QString part_name, QString text, QString comment_starting_delimiter, QString suffix);

    void run() override;

   signals:
    void completed(QString error);

   private:
    QString m_path;
    QString m_part_name;
    QString m_text;
    QString m_comment_starting_delimiter;
    QString m_suffix;
};
}  // namespace ORNL
