#ifndef PART_CONTROL_TREE_WIDGET_H_
#define PART_CONTROL_TREE_WIDGET_H_

// Qt
#include <QTreeWidget>

// Local
#include "widgets/part_widget/part_control/part_control_tree_item.h"

namespace ORNL {
    /*!
     * \brief Simple subclass to allow custom selection/drop behavior.
     */
    class PartControlTreeWidget : public QTreeWidget {
        Q_OBJECT
        public:
            //! \brief Constructor
            //! \param parent: A widget.
            PartControlTreeWidget(QWidget* parent = nullptr);

        public slots:
            //! \brief When all parts are selected, ensure that only top level objects are selected.
            void selectAll() override;

        signals:
            //! \brief When a drop ends, signal that parenting may have changed.
            void dropEnded();

        protected slots:
            //! \brief Catches when a drop occurs.
            void dropEvent(QDropEvent *event) override;

            //! \brief When the selection changes, ensure that only top level objects are selected.
            void selectionChanged(const QItemSelection &selected, const QItemSelection &deselected) override;
    };
}

#endif // PART_CONTROL_TREE_WIDGET_H_
