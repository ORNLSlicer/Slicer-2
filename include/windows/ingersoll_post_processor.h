#if 0
#ifndef INGERSOLLPOSTPROCESSOR_H
#define INGERSOLLPOSTPROCESSOR_H

#include <QWidget>
#include <QGridLayout>
#include <QLabel>
#include <QIcon>
#include <QPushButton>
#include <QStatusBar>

namespace ORNL
{
    /*!
     * \class IngersollPostProcessor
     * \brief Tool Ingersoll Post-Processor Window
     */
    class IngersollPostProcessor : public QWidget
    {
    public:
        //! \brief Constructor
        IngersollPostProcessor(QWidget* parent);

        //! \brief Destructor
        ~IngersollPostProcessor();
    signals:

    private slots:
        //! \brief Open a file dialog to choose a file to be post-processed
        void openFile2Process();

    protected:
        QWidget *parentWindow;
        void closeEvent(QCloseEvent *event);

    private:
        QString savedPath;
        QGridLayout *m_layout;
        QStatusBar *m_statusbar;
        QPushButton *m_openFile2Process;

        void setupEvents();
        void processFile();
        bool processFile(QString input, QString output);
    };
}

#endif // INGERSOLLPOSTPROCESSOR_H
#endif
