#include "graphics/view/preview_view.h"

// Local
#include "graphics/objects/grid_object.h"
#include "graphics/objects/cube/plane_object.h"
#include "graphics/objects/part_object.h"

namespace ORNL {
    PreviewView::PreviewView() {
        // NOP
    }

    void PreviewView::setPart(QSharedPointer<Part> p) {
        this->setIsoView();

        this->removeObject(m_gop);

        m_gop.reset();
        m_gop = QSharedPointer<PartObject>::create(this, p);

        m_gop->translate(QVector3D(0, 0, -m_gop->minimum().z()));

        this->addObject(m_gop);

        this->update();
    }

    void PreviewView::clearPart() {
        this->removeObject(m_gop);

        m_gop.reset();

        this->update();
    }

    void PreviewView::showSlicingPlane(bool show) {
        if (m_gop.isNull()) return;

        m_gop->plane()->setHidden(!show);

        this->update();
    }

    void PreviewView::setSlicingPlaneRotation(Angle pitch, Angle yaw, Angle roll) {
        if (m_gop.isNull()) return;

        m_gop->plane()->setLockedRotationAngle(pitch, yaw, roll);

        this->update();
    }

    void PreviewView::initView() {
        this->setIsoView();

        int lp_len = 1000;

        m_grid = QSharedPointer<GridObject>::create(this, lp_len, lp_len, lp_len / 400, lp_len / 400);

        m_plane = QSharedPointer<PlaneObject>::create(this, lp_len, lp_len, QColor(128, 128, 128, 102));
        m_plane->setUnderneath(true);

        this->addObject(m_grid);
        this->addObject(m_plane);
    }

    void PreviewView::handleLeftClick(QPointF mouse_ndc_pos) {
        // NOP
    }

} // namespace ORNL
