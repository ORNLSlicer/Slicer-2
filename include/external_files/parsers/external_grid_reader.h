#ifndef EXTERNALGRIDREADER_H
#define EXTERNALGRIDREADER_H

// Qt
#include <QThread>
#include <QString>
#include <QVector>

#include "external_files/external_grid.h"
#include "exceptions/exceptions.h"

namespace ORNL
{
    /*!
     * \class ExternalGridReader
     * \brief Threaded class that provides processing for grid structure.  Currently grid
     * represent path densities based on stress simulation
     */
    class ExternalGridReader : public QThread {

        Q_OBJECT
        public:
            //! \brief Constructor
            //! \param filename: Name of file to load
            ExternalGridReader(QString file);

            //! \brief Function that is run when start is called on this thread.
            void run() override;

        signals:

            //! \brief Signals current completion percentage
            //! \param value: current percentage complete
            void statusUpdate(int value);

            //! \brief Signals failure to parse grid file
            //! \param msg: Error message
            void gridFailed(QString msg);

            //! \brief Signals that processing of the grid is complete
            //! \param gridInfo: Grid of path percentages indexed by Z with a 2D matrix for each layer
            //! as well as the min values and step size for each dimension
            void gridFileProcessed(ExternalGridInfo gridInfo);

        private:

            //! \brief Filename.
            QString m_filename;

    };  // class ExternalGridReader
}  // namespace ORNL
#endif  // EXTERNALGRIDREADER_H
