#ifndef AUTOORIENTWIDGET_H
#define AUTOORIENTWIDGET_H

// Qt
#include <QWidget>
#include <QVBoxLayout>
#include <QtCharts>
#include <QtCharts/QChartView>

// Local
#include "part/part.h"
#include "geometry/mesh/advanced/auto_orientation.h"
#include "widgets/part_widget/model/part_meta_model.h"

Q_DECLARE_METATYPE(QSharedPointer<ORNL::Part>);

namespace ORNL
{
    //! \class AutoOrientWidget
    //! \brief a widget to display data from and control the auto orient algorithm
    class AutoOrientWidget : public QWidget
    {
        Q_OBJECT
        public:
            //! \brief Contructor
            //! \param meta the part meta model that needs to be tracked
            //! \param parent optional parent pointer
            explicit AutoOrientWidget(QSharedPointer<PartMetaModel> meta, QWidget *parent = nullptr);

        public slots:
            //! \brief widgets starts tracking a new part
            //! \param pm the part meta
            void addPart(QSharedPointer<PartMetaItem> pm);

            //! \brief widget stops tracking a part
            //! \param pm the part meta
            void removePart(QSharedPointer<PartMetaItem> pm);

        private:
            //! \brief runs the algorithm, called by the run button
            void run();

            //! \brief sets up sub widgets
            void setupWidgets();

            //! \brief enables/ disables the widget
            //! \param status what to do
            void enable(bool status);

            //! \brief clears the chart content
            void clearChart();

            //! \brief finds a part meta item by its part name
            //! \param name the name of the part
            //! \return the part meta item or nullptr
            QSharedPointer<PartMetaItem> findPartMetaByName(const QString& name);

            //! \brief the object that does runs the algorithm
            AutoOrientation* m_orienter;

            //! \brief the base layout of this widget
            QVBoxLayout* m_layout;

            //! \brief widget used to hold the chart
            QChartView * m_chart_view;

            //! \brief the chart
            QChart* m_chart;

            //! \brief series used to show data this is NOT optimal
            QScatterSeries* m_not_picked_series;

            //! \brief series used to show data this is optimal
            QScatterSeries* m_picked_series;

            //! \brief group box to hold the chart
            QGroupBox* m_chart_group;

            //! \brief group box to hold the options
            QGroupBox* m_options_group;

            //! \brief combobox for selecting the part
            QComboBox* m_part_combobox;

            QSharedPointer<PartMetaModel> m_meta;
    };
}

#endif // AUTOORIENTWIDGET_H
