#pragma once

// Qt
#include <QGraphicsView>

// Qt Forward
class QLabel;

namespace ORNL {
    class ScrollableGraphicsView : public QGraphicsView {
        Q_OBJECT
        public:
            ScrollableGraphicsView(QWidget* parent = nullptr);

            void setScrollFactor(double scroll_factor);
            double scrollFactor();

        protected slots:
            void resizeEvent(QResizeEvent* event);

            void wheelEvent(QWheelEvent* event);

            void mouseMoveEvent(QMouseEvent* event);

        private:
            double m_scroll_factor;

            QLabel* m_pos_label;
    };
}
