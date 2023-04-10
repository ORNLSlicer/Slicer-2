#ifndef REMOTE_CONNECTIVITY_H
#define REMOTE_CONNECTIVITY_H

#include <QTabWidget>
#include <QSpinBox>
#include <QTableWidget>
#include <QPushButton>
#include <QCheckBox>
#include <QLineEdit>
#include <QGroupBox>

#include "utilities/qt_json_conversion.h"
#include "utilities/enums.h"

namespace ORNL
{
    /*!
     * \class RemoteConnectivity
     * \brief Window that allows users to export gcode and project information
     */
    class RemoteConnectivity : public QTabWidget
    {
        Q_OBJECT
        public:
            //! \brief Default Constructor
            explicit RemoteConnectivity(QWidget* parent);

            //! \brief Destructor
            ~RemoteConnectivity();

            //! \brief Set initial config when program starts
            //! \param filename: Path to config file
            void setConfig(QString filename);

        protected:
            //! \brief Close event override to reset focus to parent widget
            void closeEvent(QCloseEvent *event);

        signals:
            //! \brief Set connectivity for each processing step
            //! \param step: processing step to set connectivity for (currently only gcode)
            //! \param toggle: whether or not transmission should occur
            void setStepConnectivity(StatusUpdateStepType step, bool toggle);

            //! \brief Start/Restart TCP server on specified port
            //! \param port: port to start server on
            void restartTcpServer(int port);

            //! \brief Signal to HttpServer to restart with new configuration
            //! \param config: json configuration for server to use
            void restartServer(fifojson config);

        public slots:
            //! \brief Receives update from HttpServer as to restart status
            //! \param success: whether or not restart was successful
            void restartComplete(bool success);

        private slots:
            //! \brief Saves config to file
            void saveConfig();

        private:
            //! \brief create first tab for TCP server
            void createTCPTab();

            //! \brief create second tab for HTTP server
            void createHTTPTab();

            //! \brief Creates json object from UI components
            //! \return Json object that represents current settings
            fifojson createJsonConfig();

            //! \brief TCP UI components
            QCheckBox* preProcessingCheckBox, computeCheckBox, postProcessingCheckBox, gcodeCheckBox, visualizationCheckBox;

            //! \brief UI components
            QSpinBox* m_port_box;
            QTableWidget* m_table;
            QPushButton* m_restart_button;

            //! \brief parent widget of this remote connectivity window
            QWidget* m_parent;

    };  // class RemoteConnectivity
}  // namespace ORNL

#endif // REMOTE_CONNECTIVITY_H
