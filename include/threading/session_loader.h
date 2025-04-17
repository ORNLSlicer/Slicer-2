#ifndef SESSIONLOADER_H
#define SESSIONLOADER_H

// Qt
#include <QThread>

// Zip
#include "zip/zip.h"

// JSON
#include <nlohmann/json.hpp>
#include "utilities/qt_json_conversion.h"

namespace ORNL {
    class MeshVertex;
    class MeshFace;
    class SessionManager;

    /*!
     * \class SessionLoader
     * \brief Saves or loads a session file in a separate thread.
     * \todo The session saving and loading needs some cleanup.
     */
    class SessionLoader : public QThread {
        Q_OBJECT
        public:
            //! \brief Constructor.
            //! \param filename: Filename for either loading or saving.
            //! \param save:     If true, the current session will be saved. If false, the current session will be loaded.
            SessionLoader(QString filename, bool save);

            //! \brief Get global settings from zip
            //! \return json from global settings for version check
            fifojson getSettingsFromZip();

            //! \brief Set global settings
            //! \param j: json to override when opening zip
            void updateSettingsJson(fifojson j);

            //! \brief Start the thread.
            void run() override;

        signals:
            //! \brief Signal that an error has occured.
            void error(QString error);

        private:
            //! \brief Saves a session.
            void saveSession();

            //! \brief Load session.
            void loadSession();

            //! \brief loads string from zip file
            //! \param zip: the zip file to open from
            //! \param key: the name of the file
            std::string loadStringFromZip(struct zip_t*, const std::string& key);

            //! \brief Filename this loader thread will work on.
            QString m_filename;

            //! \brief If enabled, this save a session rather than load it.
            bool m_save;

            //! \brief json to override global when opening zip
            fifojson m_new_json;

    };  // class SessionLoader
}  // namespace ORNL
#endif  // SESSIONLOADER_H
