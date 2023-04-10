#ifndef FIELD_H
#define FIELD_H

#include <cmath>
#include <QEnableSharedFromThis>
#include <QFile>
#include <QSharedPointer>
#include <QVector>

#include "graph.h"

namespace ORNL
{

    #define MAX_KEY_LENGTH 1000
    /*!
     * \class Field
     * \brief It is used to assign field values like stress to the graph
     */
    class Field
    {
    public:
        Field();

        //!
        //! \brief loadfieldfile To Read fieldvalue to Field Object
        //! \param file_path Location of file.
        //!

        void loadfieldfile(QString file_path);

        //!
        //! \brief getField Get the max value of the field belongs to the box contaning that circle
        //! \param circle
        //! \return Maximum field Value within the box
        //!

        double getField(QSharedPointer<Circle> circle);

        //!
        //! \brief selectVerticesOverFieldThreshold Select vertices having field value more than
        //! threshold value
        //! \param graph
        //! \param threshold Minimum field value for vertices selected.
        //! \return Set of Vertices selected with minimum threshold value.
        //!

        QSet< QSharedPointer<Vertex> > selectVerticesOverFieldThreshold(QSharedPointer<Graph> graph, double threshold);
        // //! \brief Normalizing field value
        //!
        //! \brief normalizeField It will normalize field values between 0 and 1.
        //!
        void normalizeField();
        // //! \brief Return fieldValue
        //!
        //! \brief fieldValue Return all field value
        //!
        QMap< QPair<int, int>, double > fieldValue();
        //!
        //! \brief fieldValue Return field value with a particular key.
        //! \param key Unique id of a field value.
        //! \return Field value associated with key.
        //!
        double fieldValue(QPair<int, int> key);
        // //! \brief Set fieldValue
        //!
        //! \brief fieldValue Assign value to field associated with key.
        //! \param key Unique id of a field value.
        //! \param value value of field
        //!
        void fieldValue(QPair<int, int> key, double value);
        //!
        //! \brief fieldValue Set all field values with field_value.
        //! \param field_value All field values.
        //!
        void fieldValue( QMap< QPair<int, int>, double> field_value);

    private:
        QMap< QPair<int, int>, double> m_field_value;

    };

}

#endif // FIELD_H
