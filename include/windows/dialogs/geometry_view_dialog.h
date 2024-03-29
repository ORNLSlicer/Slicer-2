#pragma once

// Qt
#include <QDialog>
#include <QHBoxLayout>
#include <QGraphicsView>

// Local
#include "widgets/scrollable_graphics_view.h"
#include "geometry/polyline.h"

namespace ORNL {
    class GeometryView : public QDialog {
        public:
            GeometryView(QWidget* parent = nullptr);

            void addPolyline(Polyline polyline, QColor color = QColor());

            void addPolygon(PolygonList polygon, QColor color = QColor());

            void jumpTo(Point location);

            void clear();

        private:
            ScrollableGraphicsView* m_polygon_view;

            QHBoxLayout* m_layout;
    };
}
