#include "widgets/auto_orient_widget.h"

// Qt
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QSlider>

// Local
#include "managers/gpu_manager.h"
#include "managers/preferences_manager.h"

namespace ORNL
{
    AutoOrientWidget::AutoOrientWidget(QSharedPointer<PartMetaModel> meta, QWidget *parent) : QWidget{parent}
    {

        setupWidgets();
        enable(m_part_combobox->count() > 0 && GPU->use());

        m_meta = meta;
        connect(m_meta.get(), &PartMetaModel::itemAddedUpdate, this, &AutoOrientWidget::addPart);
        connect(m_meta.get(), &PartMetaModel::itemRemovedUpdate, this, &AutoOrientWidget::removePart);
    }

    void AutoOrientWidget::addPart(QSharedPointer<PartMetaItem> pm)
    {
        auto part = pm->part();

        // Add option to combo box
        if(dynamic_cast<ClosedMesh*>(part->rootMesh().get()) != nullptr) // Only add closed meshes
            m_part_combobox->addItem(part->name());

        enable(m_part_combobox->count() > 0 && GPU->use());
    }

    void AutoOrientWidget::removePart(QSharedPointer<PartMetaItem> pm)
    {
        auto part = pm->part();

        auto index = m_part_combobox->findText(part->name());
        if(index != -1)
            m_part_combobox->removeItem(index);

        enable(m_part_combobox->count() > 0 && GPU->use());
    }

    void AutoOrientWidget::run()
    {
        clearChart();
        Plane plane;

        QSharedPointer<PartMetaItem> selected_pm = findPartMetaByName(m_part_combobox->currentText());

        if(selected_pm == nullptr) // Part not count
            return;

        auto part = selected_pm->part();
        auto mesh = dynamic_cast<ClosedMesh*>(part->rootMesh().get());

        // Copy mesh and apply current transform. We do this to a copy to not apply changes to the real copy
        ClosedMesh copy_of_mesh(*mesh);
        copy_of_mesh.setTransformation(selected_pm->transformation());

        m_orienter = new AutoOrientation(copy_of_mesh, plane);
        m_orienter->orient();

        auto distance_unit = PM->getDistanceUnit();

        auto best_pick = m_orienter->getRecommendedOrientation();

        double best_scaled_area = best_pick.area() / (distance_unit * distance_unit)();
        double best_scaled_volume = best_pick.support_volume() / (distance_unit * distance_unit * distance_unit)();
        m_picked_series->append(best_scaled_area, best_scaled_volume);

        auto results = m_orienter->getResults();
        for(auto result : results)
        {
            if(result.plane == best_pick.plane && result.build_vector == best_pick.build_vector)
                continue;

            double scaled_area = result.area() / (distance_unit * distance_unit)();
            double scaled_volume = result.support_volume() / (distance_unit * distance_unit * distance_unit)();

            m_not_picked_series->append(scaled_area, scaled_volume);
        }

        m_chart->removeSeries(m_picked_series);
        m_chart->addSeries(m_picked_series);
        m_chart->removeSeries(m_not_picked_series);
        m_chart->addSeries(m_not_picked_series);

        m_chart->createDefaultAxes();

        // Scale axis to be 10% than the data range to fit nicely. Also ensure that axis ranges do not go below zero
        auto axes = m_chart->axes();
        if(axes.size() == 2)
        {
            auto x_axis = dynamic_cast<QValueAxis*>(axes[0]);
            x_axis->setTitleText("Surface area");
            x_axis->setTitleFont(QFont("Times", 12, QFont::DemiBold));
            x_axis->setTitleVisible(true);
            x_axis->setLabelsFont(QFont("Times", 10, QFont::Normal));
            x_axis->setLabelFormat("%.2f " + PM->getDistanceUnitText() + "<sup>2</sup>");

            auto x_range = x_axis->max() - x_axis->min();
            x_axis->setMax(x_axis->max() + (x_range * 0.1));
            x_axis->setMin(x_axis->min() - (x_range * 0.1));

            if(x_axis->min() < 0.0)
                x_axis->setMin(0);

            auto y_axis = dynamic_cast<QValueAxis*>(axes[1]);
            y_axis->setTitleText("Support volume");
            y_axis->setTitleFont(QFont("Times", 12, QFont::DemiBold));
            y_axis->setTitleVisible(true);
            y_axis->setLabelsFont(QFont("Times", 10, QFont::Normal));
            y_axis->setLabelFormat("%.2g " + PM->getDistanceUnitText() + "<sup>3</sup>");

            auto y_range = y_axis->max() - y_axis->min();
            y_axis->setMax(y_axis->max() + (y_range * 0.1));
            y_axis->setMin(y_axis->min() - (y_range * 0.1));

            if(y_axis->min() < 0.0)
                y_axis->setMin(0);
        }

        m_chart_view->repaint();
    }

    void AutoOrientWidget::setupWidgets()
    {
        m_layout = new QVBoxLayout();

        QLabel* gpu_warning_label = new QLabel("Auto orientation requires a compatible GPU to be installed in the system and \"Enable GPU Acceleration\" to be set under optimization settings. You can check \"Help -> About ORNL Slicer 2\" for GPU compatibility or reference the User's Manual for further information.");
        gpu_warning_label->setWordWrap(true);
        gpu_warning_label->setVisible(!GPU->use());
        m_layout->addWidget(gpu_warning_label);

        m_options_group = new QGroupBox("Options");
        auto options_layout = new QHBoxLayout();
        m_options_group->setLayout(options_layout);

        // Part drop-down selector
        QLabel* part_label = new QLabel("Part:");
        m_part_combobox = new QComboBox();
        m_part_combobox->setMinimumWidth(200);

        connect(m_part_combobox, &QComboBox::currentTextChanged, this, [this](const QString& item)
        {
            clearChart();
        });

        options_layout->addWidget(part_label);
        options_layout->addWidget(m_part_combobox);

        options_layout->addStretch();

        QPushButton* run_btn = new QPushButton("Run");
        connect(run_btn, &QPushButton::clicked, this, &AutoOrientWidget::run);

        options_layout->addWidget(run_btn);

        // Chart series
        m_not_picked_series = new QScatterSeries();
        m_not_picked_series->setName("Other Options");
        m_not_picked_series->setMarkerShape(QScatterSeries::MarkerShapeCircle);
        m_not_picked_series->setColor(QColor::fromRgb(255, 0, 0));
        m_not_picked_series->setMarkerSize(8.0);
        m_not_picked_series->setBorderColor(Qt::transparent);

        m_picked_series = new QScatterSeries();
        m_picked_series->setName("Recommended");
        m_picked_series->setMarkerShape(QScatterSeries::MarkerShapeRectangle);
        m_picked_series->setColor(QColor::fromRgb(0, 255, 0));
        m_picked_series->setMarkerSize(12.0);
        m_picked_series->setBorderColor(Qt::transparent);

        // Setup chart object
        m_chart = new QChart();

        // Setup chart view with new chart
        m_chart_view = new QChartView(m_chart);
        m_chart_view->setRenderHint(QPainter::Antialiasing);
        m_chart_view->setRubberBand(QChartView::RectangleRubberBand);
        auto distance_unit = PM->getDistanceUnit();
        connect(m_not_picked_series, &QXYSeries::clicked, this, [this, distance_unit](const QPointF &point)
        {
            QSharedPointer<PartMetaItem> selected_pm = findPartMetaByName(m_part_combobox->currentText());

            if(selected_pm == nullptr) // Part not count
                return;


            double micron_area = point.x() * distance_unit() * distance_unit();
            double micron_volume = point.y() * distance_unit() * distance_unit() * distance_unit();

            auto picked_orentation = m_orienter->getOrientationForValues(micron_area, micron_volume);
            selected_pm->setRotation(AutoOrientation::GetRotationForOrientation(picked_orentation));

            // Drop the part to the bed
            float z_sub = Constants::Limits::Maximums::kMaxFloat;
            // Find the actual minimum on the object.
            for (Triangle t : selected_pm->graphicsPart()->triangles())
            {
                for (uint i = 0; i < 3; i++)
                {
                    QVector3D pt = t[i];

                    if (pt.z() < z_sub) z_sub = pt.z();
                }
            }

            QVector3D trans = selected_pm->translation();
            trans.setZ(trans.z() - z_sub);

            selected_pm->setTranslation(trans);
        });

        connect(m_picked_series, &QXYSeries::clicked, this, [this, distance_unit](const QPointF &point)
        {
            QSharedPointer<PartMetaItem> selected_pm = findPartMetaByName(m_part_combobox->currentText());

            if(selected_pm == nullptr) // Part not count
                return;

            double micron_area = point.x() * distance_unit() * distance_unit();
            double micron_volume = point.y() * distance_unit() * distance_unit() * distance_unit();

            auto picked_orentation = m_orienter->getOrientationForValues(micron_area, micron_volume);
            selected_pm->setRotation(AutoOrientation::GetRotationForOrientation(picked_orentation));

            // Drop the part to the bed
            float z_sub = Constants::Limits::Maximums::kMaxFloat;

            // Find the actual minimum on the object.
            for (Triangle t : selected_pm->graphicsPart()->triangles())
            {
                for (uint i = 0; i < 3; i++)
                {
                    QVector3D pt = t[i];

                    if (pt.z() < z_sub) z_sub = pt.z();
                }
            }

            QVector3D trans = selected_pm->translation();
            trans.setZ(trans.z() - z_sub);

            selected_pm->setTranslation(trans);
        });

        // Setup chart features
        m_chart->setTitle("Support Volume vs. First Layer Surface Area");
        m_chart->setTitleFont(QFont("Times", 13 , QFont::Bold));
        m_chart->setFont(QFont("Times", 11, QFont::Normal));
        m_chart->setDropShadowEnabled(false);
        m_chart->legend()->setFont(QFont("Times", 12, QFont::Normal));

        // Create a group box and add chart
        m_chart_group = new QGroupBox("Results");
        auto chart_layout = new QVBoxLayout();

        QLabel* directions = new QLabel("After running, double-click data points below to rotate the part in the view.");
        chart_layout->addWidget(directions);
        chart_layout->addWidget(m_chart_view);

        QHBoxLayout* chart_controls_layout = new QHBoxLayout();

        QPushButton* zoom_in_btn = new QPushButton(QIcon(":/icons/magnify_plus_black.png"),"");
        zoom_in_btn->setToolTip("Zoom In");
        connect(zoom_in_btn, &QPushButton::clicked, this, [this](){
            m_chart->zoomIn();
        });
        chart_controls_layout->addWidget(zoom_in_btn);

        QPushButton* zoom_out_btn = new QPushButton(QIcon(":/icons/magnify_minus_black.png"),"");
        zoom_out_btn->setToolTip("Zoom Out");
        connect(zoom_out_btn, &QPushButton::clicked, this, [this](){
            m_chart->zoomOut();
        });
        chart_controls_layout->addWidget(zoom_out_btn);

        QPushButton* reset_zoom_btn = new QPushButton(QIcon(":/icons/magnify_reset_black.png"),"");
        reset_zoom_btn->setToolTip("Reset Zoom");
        connect(reset_zoom_btn, &QPushButton::clicked, this, [this](){
            m_chart->zoomReset();
        });
        chart_controls_layout->addWidget(reset_zoom_btn);

        QPushButton* save_chart_btn = new QPushButton(QIcon(":/icons/save_black.png"),"");
        save_chart_btn->setToolTip("Save image");
        connect(save_chart_btn, &QPushButton::clicked, this, [this](){
            QString file_name = "chart.jpg";
            file_name = QFileDialog::getSaveFileName(this,"...","..."," (*.jpg)");
            if(file_name != "")
            {
                QPixmap picture;
                picture = m_chart_view->grab();
                picture.save(file_name);
            }
        });
        chart_controls_layout->addWidget(save_chart_btn);

        chart_controls_layout->addStretch();

        chart_layout->addLayout(chart_controls_layout);
        m_chart_group->setLayout(chart_layout);

        m_layout->addWidget(m_options_group);
        m_layout->addWidget(m_chart_group);
        this->setLayout(m_layout);
    }

    void AutoOrientWidget::enable(bool status)
    {
        m_options_group->setEnabled(status);
        m_chart_group->setEnabled(status);
    }

    void AutoOrientWidget::clearChart()
    {
        m_picked_series->clear();
        m_not_picked_series->clear();

        auto chart_series = m_chart->series();
        if(chart_series.contains(m_picked_series))
            m_chart->removeSeries(m_picked_series);

        if(chart_series.contains(m_not_picked_series))
            m_chart->removeSeries(m_not_picked_series);
        m_chart->createDefaultAxes();

        auto axes = m_chart->axes();
        if(axes.size() == 2)
        {
            axes[0]->setTitleText("Surface area (" + PM->getDistanceUnitText() + "^2)");
            axes[0]->setTitleVisible(true);

            axes[1]->setTitleText("Support volume (" + PM->getDistanceUnitText() + "^3)");
            axes[1]->setTitleVisible(true);
        }

        m_chart_view->repaint();
    }

    QSharedPointer<PartMetaItem> AutoOrientWidget::findPartMetaByName(const QString& name)
    {
        QSharedPointer<PartMetaItem> selected_pm = nullptr;
        for(auto pm : m_meta->items())
            if(pm->part()->name() == m_part_combobox->currentText())
                selected_pm = pm;
        return selected_pm;
    }
}
