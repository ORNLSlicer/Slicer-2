#include "widgets/part_widget/part_control/part_control_tree_widget.h"

// Qt
#include <QtGlobal>
#include <QStack>

namespace ORNL {

    PartControlTreeWidget::PartControlTreeWidget(QWidget* parent) : QTreeWidget(parent) {
        // NOP
    }

    void PartControlTreeWidget::selectAll() {
        QList<QTreeWidgetItem*> tl = this->findItems("*", Qt::MatchWildcard | Qt::MatchRecursive);
        for (QTreeWidgetItem* item : tl) {
            if (item->parent()) item->setSelected(false);
            else item->setSelected(true);
        }
    }

    /*
    void PartControlTreeWidget::mouseMoveEvent(QMouseEvent* event) {
        // From Qt source - Without dragging to select.
        Q_D(QTreeWidget);
        QPoint topLeft;
        QPoint bottomRight = event->pos();
        if (state() == ExpandingState || state() == CollapsingState)
            return;

        if (state() == DraggingState) {
            topLeft = d->pressedPosition - d->offset();
            if ((topLeft - bottomRight).manhattanLength() > QApplication::startDragDistance()) {
                d->pressedIndex = QModelIndex();
                startDrag(d->model->supportedDragActions());
                setState(NoState); // the startDrag will return when the dnd operation is done
                stopAutoScroll();
            }
            return;
        }

        QPersistentModelIndex index = indexAt(bottomRight);
        QModelIndex buddy = d->model->buddy(d->pressedIndex);
        if ((state() == EditingState && d->hasEditor(buddy))
            || edit(index, NoEditTriggers, event))
            return;
        if (d->selectionMode != SingleSelection)
            topLeft = d->pressedPosition - d->offset();
        else
            topLeft = bottomRight;
        d->checkMouseMove(index);

        if (d->pressedIndex.isValid()
            && d->dragEnabled
            && (state() != DragSelectingState)
            && (event->buttons() != Qt::NoButton)
            && !d->selectedDraggableIndexes().isEmpty()) {
                setState(DraggingState);
                return;
        }
    }
    */

    void PartControlTreeWidget::dropEvent(QDropEvent* event) {
        this->QTreeWidget::dropEvent(event);

        emit dropEnded();
    }

    void PartControlTreeWidget::selectionChanged(const QItemSelection& selected, const QItemSelection& deselected) {
        this->QTreeWidget::selectionChanged(selected, deselected);

        // If a call occurs with no new selected items, we can bail immediately.
        QModelIndexList l = selected.indexes();
        if (l.empty()) return;

        QTreeWidgetItem* item = this->itemFromIndex(l.first());

        // New selection cannot have parents or children selected.
        QTreeWidgetItem* par = item->parent();

        while (par) {
            if (par->isSelected()) {
                par->setSelected(false);
            }

            par = par->parent();
        }

        QStack<QTreeWidgetItem*> queue;
        for (int i = 0; i < item->childCount(); i++) {
            queue.push(item->child(i));
        }

        while (!queue.empty()) {
            QTreeWidgetItem* curr_item = queue.pop();

            for (int i = 0; i < curr_item->childCount(); i++) {
                queue.push(curr_item->child(i));
            }

            if (curr_item->isSelected()) {
                curr_item->setSelected(false);
            }
        }
    }
}
