#include "graphics/view/part_view.h"

// Qt
#include <QStack>
#include <QToolTip>
#include <QMessageBox>

// Local
#include "graphics/graphics_object.h"
#include "graphics/objects/part_object.h"
#include "graphics/objects/text_object.h"
#include "graphics/objects/cube/plane_object.h"
#include "graphics/objects/arrow_object.h"
#include "graphics/objects/axes_object.h"
#include "graphics/objects/grid_object.h"
#include "graphics/objects/cube_object.h"
#include "graphics/objects/sphere_object.h"
#include "graphics/objects/printer/printer_object.h"
#include "graphics/objects/printer/cartesian_printer_object.h"
#include "graphics/objects/printer/cylindrical_printer_object.h"
#include "graphics/objects/printer/toroidal_printer_object.h"

#include "graphics/support/shape_factory.h"
#include "managers/settings/settings_manager.h"
#include "managers/preferences_manager.h"
#include "managers/session_manager.h"
#include "part/part.h"
#include "graphics/support/part_picker.h"
#include "utilities/debugutils.h"
#include "utilities/mathutils.h"
#include "widgets/part_widget/model/part_meta_model.h"
#include "widgets/part_widget/right_click_menu.h"

namespace ORNL
{
    PartView::PartView(QSharedPointer<SettingsBase> sb)
    {
        m_menu = new RightClickMenu(this);
        m_sb = sb;
    }

    void PartView::setModel(QSharedPointer<PartMetaModel> m)
    {
        m_model = m;

        QObject::connect(m_model.get(), &PartMetaModel::itemAddedUpdate, this, &PartView::modelAdditionUpdate);
        QObject::connect(m_model.get(), &PartMetaModel::itemReloadUpdate, this, &PartView::modelReloadUpdate);
        QObject::connect(m_model.get(), &PartMetaModel::itemRemovedUpdate, this, &PartView::modelRemovalUpdate);
        QObject::connect(m_model.get(), &PartMetaModel::selectionUpdate, this, &PartView::modelSelectionUpdate);
        QObject::connect(m_model.get(), &PartMetaModel::parentingUpdate, this, &PartView::modelParentingUpdate);
        QObject::connect(m_model.get(), &PartMetaModel::transformUpdate, this, &PartView::modelTranformUpdate);
        QObject::connect(m_model.get(), &PartMetaModel::visualUpdate, this, &PartView::modelVisualUpdate);

        QObject::connect(CSM.get(), &SessionManager::partReloaded, this, &PartView::modelReloadUpdate);
    }

    QList<QSharedPointer<Part>> PartView::floatingParts()
    {
        QList<QSharedPointer<Part>> result;
        QVector3D printerFloor = m_printer->printerCenter();

        for (auto& gop : m_part_objects) {
            if(!MathUtils::glEquals(gop->minimum().z(), printerFloor.z())) {
                result.push_back(gop->part());
            }
        }

        return result;
    }

    QList<QSharedPointer<Part>> PartView::externalParts()
    {
        QList<QSharedPointer<PartObject>> ret = m_printer->externalParts();
        QList<QSharedPointer<Part>> result;

        for (auto& gop : ret) {
            result.append(gop->part());
        }

        return result;
    }

    void PartView::showLabels(bool show)
    {
        for (auto& gop : m_part_objects) {
            gop->label()->setHidden(!show);
        }

        m_state.names_shown = show;

        this->update();
    }

    void PartView::showSlicingPlanes(bool show)
    {
        for (auto& gop : m_part_objects) {
            gop->plane()->setHidden(!show);
        }

        m_state.planes_shown = show;

        this->update();
    }

    void PartView::showOverhang(bool show)
    {
        for (auto& gop : m_part_objects) {
            gop->showOverhang(show);
        }

        m_state.overhangs_shown = show;

        this->update();
    }

    void PartView::showSeams(bool show)
    {
        m_printer->setSeamsHidden(!show);

        m_state.seams_shown = show;

        this->update();
    }

    void PartView::setupAlignment(QVector3D plane)
    {
        m_state.align_plane_norm = plane;
        this->setCursor(Qt::PointingHandCursor);

        m_state.aligning = true;
    }

    void PartView::centerPart(QString name)
    {
        auto gop = this->findObject(name);

        if(!gop.isNull())
        {
            this->centerPart(gop);

            this->blockModel();
            m_model->lookupByGraphic(gop)->setTranslation(gop->translation());
            this->permitModel();

            this->postTransformCheck();
            this->update();
        }
    }

    void PartView::centerPart(QSharedPointer<PartObject> gop)
    {
        QVector3D printer_center = m_printer->printerCenter();
        gop->translateAbsolute(QVector3D(printer_center.x(), printer_center.y(), gop->translation().z()));
    }

    void PartView::dropPart(QSharedPointer<PartObject> gop)
    {
        float z_sub = Constants::Limits::Maximums::kMaxFloat;

        // Find the actual minimum on the object.
        for (Triangle t : gop->triangles())
        {
            for (uint i = 0; i < 3; i++)
            {
                QVector3D pt = t[i];

                if (pt.z() < z_sub) z_sub = pt.z();
            }
        }

        QVector3D trans = gop->translation();
        z_sub -= m_printer->printerCenter().z();
        trans.setZ(trans.z() - z_sub);

        gop->translateAbsolute(trans, true);
    }

    void PartView::shiftPart(QSharedPointer<PartObject> gop)
    {
        // Object is now in printer center. See if any other objects intersect and shift if necessary.
        restart_check: for (auto& egop : m_part_objects)
        {
            if (egop->doesMBBIntersect(gop))
            {
                gop->translate(QVector3D(0.5, 0, 0));
                goto restart_check;
            }
        }
    }

    void PartView::dropSelectedParts()
    {
        for (auto& gop : m_selected_objects)
        {
            dropPart(gop);

            this->blockModel();
            m_model->lookupByGraphic(gop)->setTranslation(gop->translation() - m_printer->minimum());
            this->permitModel();
        }

        this->postTransformCheck();
        this->update();
    }

    void PartView::updatePrinterSettings(QSharedPointer<SettingsBase> sb)
    {
        this->resetCamera();

        m_sb = sb;

        BuildVolumeType buildVolume = static_cast<BuildVolumeType>(m_sb->setting<int>(Constants::PrinterSettings::Dimensions::kBuildVolumeType));

        QSharedPointer<PrinterObject> new_printer;

        switch (buildVolume) {
            case ORNL::BuildVolumeType::kRectangular:
                if (m_printer.dynamicCast<CartesianPrinterObject>().isNull()) {
                    new_printer = QSharedPointer<CartesianPrinterObject>::create(this, m_sb, false);
                }
                break;
            case ORNL::BuildVolumeType::kCylindrical:
                if (m_printer.dynamicCast<CylindricalPrinterObject>().isNull()) {
                    new_printer = QSharedPointer<CylindricalPrinterObject>::create(this, m_sb, false);
                }
                break;
            case ORNL::BuildVolumeType::kToroidal:
                if (m_printer.dynamicCast<ToroidalPrinterObject>().isNull()) {
                    new_printer = QSharedPointer<ToroidalPrinterObject>::create(this, m_sb, false);
                }
                break;
        }

        if (!new_printer.isNull())
        {
            this->resetCamera();

            // Orphan all old printer children and give them to the new printer.
            for (auto& go : m_printer->children())
            {
                auto gop = go.dynamicCast<PartObject>();

                if (!gop.isNull())
                {
                    m_printer->orphanChild(gop);
                    new_printer->adoptChild(gop);
                }
            }

            m_printer->orphanChild(m_low_plane);
            new_printer->adoptChild(m_low_plane);

            this->removeObject(m_printer);
            this->addObject(new_printer);

            new_printer->setSeamsHidden(!m_state.seams_shown);

            m_printer = new_printer;
        }
        else
        {
            m_printer->updateFromSettings(m_sb);
        }

        QVector3D max = m_printer->maximum();
        QVector3D min = m_printer->minimum();

        // For low plane, extend distance by 20% and select max.
        float width_diff =  (max.x() * 1.2) - max.x();
        float length_diff = (max.y() * 1.2) - max.y();

        float diff = 0;

        if (width_diff > length_diff) diff = width_diff;
        else diff = length_diff;

        float length = (max.x() - min.x()) + (2 * diff);
        float width =  (max.y() - min.y()) + (2 * diff);

        m_low_plane->updateDimensions(length, width, length / 50, width / 50);
        m_low_plane->translateAbsolute(m_printer->printerCenter());
        m_low_plane->translate(QVector3D(0, 0, -0.1));

        m_camera->setDefaultZoom(m_printer->getDefaultZoom());

        m_focus->updateDimensions(m_printer->getDefaultZoom() * 0.1);

        this->postTransformCheck();
        this->update();
    }

    void PartView::updateOptimizationSettings(QSharedPointer<SettingsBase> sb)
    {
        m_sb = sb;

        m_printer->updateFromSettings(m_sb);

        this->update();
    }

    void PartView::updateOverhangSettings(QSharedPointer<SettingsBase> sb)
    {
        m_sb = sb;

        for (auto& gop : m_part_objects) {
            gop->setOverhangAngle(m_sb->setting<Angle>(Constants::ProfileSettings::Support::kThresholdAngle));
        }

        this->update();
    }

    void PartView::updateSlicingSettings(QSharedPointer<SettingsBase> sb)
    {
        m_sb = sb;

        Angle slicing_plane_pitch = m_sb->setting<Angle>(Constants::ExperimentalSettings::SlicingAngle::kStackingDirectionPitch);
        Angle slicing_plane_yaw   = m_sb->setting<Angle>(Constants::ExperimentalSettings::SlicingAngle::kStackingDirectionYaw);
        Angle slicing_plane_roll  = m_sb->setting<Angle>(Constants::ExperimentalSettings::SlicingAngle::kStackingDirectionRoll);

        for (auto& gop : m_part_objects) {
            gop->plane()->setLockedRotationAngle(slicing_plane_pitch, slicing_plane_yaw, slicing_plane_roll);
        }

        // Changing the plane affects the overhang angle
        updateOverhangSettings(sb);

        this->update();
    }

    void PartView::initView()
    {
        BuildVolumeType buildVolume = static_cast<BuildVolumeType>(m_sb->setting<int>(Constants::PrinterSettings::Dimensions::kBuildVolumeType));

        switch (buildVolume)
        {
            case ORNL::BuildVolumeType::kRectangular:
                m_printer = QSharedPointer<CartesianPrinterObject>::create(this, m_sb, false);
                break;
            case ORNL::BuildVolumeType::kCylindrical:
                m_printer = QSharedPointer<CylindricalPrinterObject>::create(this, m_sb, false);
                break;
            case ORNL::BuildVolumeType::kToroidal:
                m_printer = QSharedPointer<ToroidalPrinterObject>::create(this, m_sb, false);
                break;
        }

        QVector3D max = m_printer->maximum();
        QVector3D min = m_printer->minimum();

        // For low plane, extend distance by 20% and select max.
        float width_diff =  (max.x() * 1.2) - max.x();
        float length_diff = (max.y() * 1.2) - max.y();

        float diff = 0;

        if (width_diff > length_diff) diff = width_diff;
        else diff = length_diff;

        float length = (max.x() - min.x()) + (2 * diff);
        float width =  (max.y() - min.y()) + (2 * diff);

        QColor c = Constants::Colors::kRed;
        c.setAlpha(102);

        m_low_plane = QSharedPointer<GridObject>::create(this, length, width, length / 50, width / 50, c);
        m_low_plane->translateAbsolute(m_printer->printerCenter());
        m_low_plane->translate(QVector3D(0, 0, -0.1));
        m_low_plane->hide();

        m_printer->adoptChild(m_low_plane);

        this->addObject(m_printer);

        m_camera->setDefaultZoom(m_printer->getDefaultZoom());
        this->resetCamera();
    }

    void PartView::handleLeftClick(QPointF mouse_ndc_pos)
    {
        auto picked_part = this->pickPart(mouse_ndc_pos, m_part_objects);
        if (!picked_part.isNull()) {
            // If currently in an alignment state, try to align.
            if (m_state.aligning) {
                Triangle picked_tri = PartPicker::pickTriangle(this->projectionMatrix(), this->viewMatrix(), mouse_ndc_pos, picked_part->triangles());

                if (!(picked_tri.a == QVector3D() && picked_tri.b == QVector3D() && picked_tri.c == QVector3D())) {
                    if (!picked_part->locked()) this->alignPart(picked_part, picked_tri, m_state.align_plane_norm);
                }

                m_state.aligning = false;
                this->postTransformCheck();
                this->setCursor(QCursor(Qt::ArrowCursor));
            }
            // If not see if we hit a selected object.
            else if (m_selected_objects.contains(picked_part)) {
                 m_state.translating = true;
            }

            return;
        }

        // Unselect all parts.
        if (!m_selected_objects.empty()) {
            this->blockModel();
            for (auto& gop : m_selected_objects) {
                gop->unselect();

                auto item = m_model->lookupByGraphic(gop);
                item->setSelected(false);
            }
            this->permitModel();

            m_selected_objects.clear();
        }

        // If we were aligning and the user clicked off of a part, reset the state.
        if (m_state.aligning) {
            m_state.aligning = false;
            this->setCursor(QCursor(Qt::ArrowCursor));
        }

        this->update();
    }

    void PartView::handleLeftDoubleClick(QPointF mouse_ndc_pos)
    {
        auto picked_part = this->pickPart(mouse_ndc_pos, m_part_objects);

        if (!picked_part.isNull()) {
            // Unselect
            if (m_selected_objects.contains(picked_part)) {
                picked_part->unselect();
                m_selected_objects.remove(picked_part);
            }
            // Select
            else {
                m_selected_objects.insert(picked_part);
                this->setCursor(QCursor(Qt::OpenHandCursor));
                m_selected_objects.subtract(picked_part->select());
                m_state.translating = true;
            }

            this->blockModel();
            for (auto& gop : m_part_objects) {
                auto item = m_model->lookupByGraphic(gop);

                if (!item->isSelected() && m_selected_objects.contains(gop)) {
                    item->setSelected(true);
                }
                else if (item->isSelected() && !m_selected_objects.contains(gop)) {
                    item->setSelected(false);
                }
            }
            this->permitModel();
        }

        this->update();
    }

    void PartView::handleLeftRelease(QPointF mouse_ndc_pos)
    {
        m_low_plane->hide();
        m_state.translate_start = QVector3D(0, 0, 0);
        m_state.part_trans_start.clear();

        this->permitModel();
        m_state.blocking = false;
        m_state.translating = false;

        this->blockModel();
        for (auto& gop : m_part_objects) {
            m_model->lookupByGraphic(gop)->setTransformation(gop->transformation());
            m_model->lookupByGraphic(gop)->setTranslation(gop->translation());
        }
        this->permitModel();

        this->postTransformCheck();
        this->update();
    }

    void PartView::handleLeftMove(QPointF mouse_ndc_pos)
    {
        if (m_selected_objects.empty()) return;
        if (!m_state.translating) return;

        if (!m_state.blocking) {
            this->blockModel();
            m_state.blocking = true;
        }

        // Translate the low plane before calculating intersection.
        if (m_state.translate_start.isNull()) {
           auto picked_part = this->pickPart(mouse_ndc_pos, m_selected_objects);

           if (!picked_part.isNull()) {
               QVector3D plane_translation = m_low_plane->translation();
               plane_translation.setZ(picked_part->minimum().z() - 0.1);

               m_low_plane->translateAbsolute(plane_translation);
               m_low_plane->show();
           }
        }

        QVector3D intersect = std::get<1>(PartPicker::pickDistanceAndIntersection(this->projectionMatrix(), this->viewMatrix(), mouse_ndc_pos, m_low_plane->triangles()));
        // If there's no intersection, there is no move.
        if (std::fabs(intersect.x()) == Constants::Limits::Maximums::kInfFloat) return;
        if (std::fabs(intersect.y()) == Constants::Limits::Maximums::kInfFloat) return;
        if (std::fabs(intersect.z()) == Constants::Limits::Maximums::kInfFloat) return;

        intersect.setZ(0);

        if (m_state.translate_start.isNull()) {
            // Save the start of this transform.
            m_state.translate_start = intersect;

            for(auto& gop : m_selected_objects) {
                m_state.part_trans_start[gop] = gop->translation();
            }
        }

        // Actual translation.
        QVector3D v = intersect - m_state.translate_start;
        for (auto& gop : m_selected_objects) {
            if (gop->locked()) {
                QToolTip::showText(QCursor::pos(), "This object is locked.", nullptr, QRect(), 300000);
                continue;
            }

            gop->translateAbsolute(v + m_state.part_trans_start[gop]);
            m_model->lookupByGraphic(gop)->setTranslation(gop->translation() - m_printer->minimum());
        }

        this->update();
    }

    void PartView::handleRightClick(QPointF mouse_ndc_pos, QPointF global_pos)
    {
        auto p = this->pickPart(mouse_ndc_pos, m_selected_objects);
        if (p.isNull() || !m_selected_objects.contains(p)) {
            this->BaseView::handleRightClick(mouse_ndc_pos, global_pos);
            return;
        }

        // We will start a rotation, but if the user lets go in under 250ms, show a right click menu instead.
        m_state.right_click_timer.start();

        m_state.rotating = true;

        m_state.rotate_start = mouse_ndc_pos;

        for (auto& gop : m_selected_objects) {
            if (gop->locked()) {
                QToolTip::showText(QCursor::pos(), "This object is locked.", nullptr, QRect(), 300000);
                continue;
            }

            m_state.part_rot_start[gop] = gop->rotation();
            gop->axes()->show();
        }
    }

    void PartView::handleRightMove(QPointF mouse_ndc_pos)
    {
        if (!m_state.rotating) {
            this->BaseView::handleRightMove(mouse_ndc_pos);
            return;
        }

        if (m_state.blocking == false) {
            this->blockModel();
            m_state.blocking = true;
        }

        QPointF delta = m_state.rotate_start - mouse_ndc_pos;

        // Actual rotation
        QVector3D r = QVector3D(delta.y() * 90, 0, -delta.x() * 90);

        QMatrix4x4 view_mtrx = this->viewMatrix();

        QVector3D right = QVector3D(view_mtrx(0,0), view_mtrx(0,1), view_mtrx(0,2));
        //QVector3D up = QVector3D(view_mtrx(1,0), view_mtrx(1,1), view_mtrx(1,2));
        QVector3D up = QVector3D(0, 0, 1);

        right.setZ(0);
        right.normalize();

        QQuaternion qr = QQuaternion::fromAxisAndAngle(right, r.x());
        qr *= QQuaternion::fromAxisAndAngle(up, r.z());

        //qDebug() << "Applied rotation" << qr.toEulerAngles();

        for (auto& gop : m_selected_objects) {
            if (gop->locked()) {
                QToolTip::showText(QCursor::pos(), "This object is locked.", nullptr, QRect(), 300000);
                continue;
            }

            m_model->lookupByGraphic(gop)->setRotation(gop->rotation(), true);

            //qDebug() << gop->name() << "rotates to" << gop->rotation().toEulerAngles() << gop->rotation();
        }

        this->update();
    }

    void PartView::handleRightRelease(QPointF mouse_ndc_pos, QPointF global_pos)
    {
        if (m_state.rotating) {
            if (m_state.right_click_timer.elapsed() < 250) {
                m_menu->show(global_pos, m_model->selectedItems());
            }

            m_state.rotating = false;
            m_state.blocking = false;
            m_state.part_rot_start.clear();
            m_state.rotate_start = QPointF();

            for (auto& gop : m_selected_objects) {
                gop->axes()->hide();

                if (gop->locked()) {
                    QToolTip::showText(QCursor::pos(), "This object is locked.", nullptr, QRect(), 300000);
                    continue;
                }

                QQuaternion r = gop->rotation();
                QVector3D er = r.toEulerAngles();

                // Snap to intervals of 15 degrees.
                er.setX(MathUtils::snap(er.x(), 15));
                er.setY(MathUtils::snap(er.y(), 15));
                er.setZ(MathUtils::snap(er.z(), 15));

                QQuaternion sr = QQuaternion::fromEulerAngles(er);

                if (std::fabs(sr.z()) >= 90) {
                    sr.setX(sr.x() + 180.f);
                    sr.setY(180.f - sr.y());
                    sr.setZ(sr.z() + 180.f);
                }

                gop->rotateAbsolute(QQuaternion::fromEulerAngles(er));
            }

            auto tmp_selected = m_selected_objects;
            for (auto& gop : tmp_selected) {
                if (gop->locked()) {
                    QToolTip::showText(QCursor::pos(), "This object is locked.", nullptr, QRect(), 300000);
                    continue;
                }

                m_model->lookupByGraphic(gop)->setRotation(gop->rotation());
            }

            this->permitModel();
            this->postTransformCheck();

            this->blockModel();
            for (auto& gop : m_part_objects) {
                if (gop->locked()) {
                    QToolTip::showText(QCursor::pos(), "This object is locked.", nullptr, QRect(), 300000);
                    continue;
                }

                m_model->lookupByGraphic(gop)->setTransformation(gop->transformation());
                m_model->lookupByGraphic(gop)->setTranslation(gop->translation() - m_printer->minimum());
            }
            this->permitModel();

            this->update();
        }
        else {
            this->BaseView::handleRightRelease(mouse_ndc_pos, global_pos);
        }
    }

    void PartView::handleMouseMove(QPointF mouse_ndc_pos)
    {
        auto picked_part = this->pickPart(mouse_ndc_pos, m_part_objects);

        QCursor c = QCursor(Qt::ArrowCursor);

        if (m_state.aligning) {
            c = QCursor(Qt::PointingHandCursor);
        }
        else if (m_selected_objects.contains(picked_part)) {
            c = QCursor(Qt::OpenHandCursor);
        }

        // Highlighting of objects.
        if (m_state.highlighted_part.isNull() && !picked_part.isNull()) {
            picked_part->highlight();
            m_state.highlighted_part = picked_part;
            this->update();
        }
        else if (!m_state.highlighted_part.isNull() && m_state.highlighted_part != picked_part) {
            m_state.highlighted_part->unhighlight();
            m_state.highlighted_part.reset();
            this->update();
        }

        this->setCursor(c);
        if (m_state.translating) {
            m_state.translating = false;
            this->postTransformCheck();
        }
    }

    void PartView::handleWheelForward(QPointF mouse_ndc_pos, float delta)
    {
        this->BaseView::handleWheelBackward(mouse_ndc_pos, delta);
        m_camera->zoom(delta);
        this->update();
    }

    void PartView::handleWheelBackward(QPointF mouse_ndc_pos, float delta)
    {
        this->BaseView::handleWheelForward(mouse_ndc_pos, delta);
        m_camera->zoom(delta);
        this->update();
    }

    void PartView::handleMidClick(QPointF mouse_ndc_pos)
    {
        this->BaseView::handleMidClick(mouse_ndc_pos);
    }

    void PartView::handleMidRelease(QPointF mouse_ndc_pos)
    {
        this->BaseView::handleMidRelease(mouse_ndc_pos);
    }

    void PartView::resetCamera()
    {
        //Reset rotation and zoom
        m_camera->reset();

        this->translateCamera(m_printer->printerCenter(), true);

        this->update(); //Need to repaint with new model matrices
    }

    void PartView::modelSelectionUpdate(QSharedPointer<PartMetaItem> pm)
    {
        QSharedPointer<PartObject> gop = pm->graphicsPart();

        // Update the selection.
        if (pm->isSelected() && !m_selected_objects.contains(gop)) {
            m_selected_objects.insert(gop);
            m_selected_objects.subtract(gop->select());
        }
        else if (!pm->isSelected() && m_selected_objects.contains(gop)){
            gop->unselect();
            m_selected_objects.remove(gop);
        }

        this->update();
    }

    void PartView::modelAdditionUpdate(QSharedPointer<PartMetaItem> pm)
    {
        QSharedPointer<PartObject> gop = pm->graphicsPart();

        // New part
        if (gop.isNull())
        {
            gop = QSharedPointer<PartObject>::create(this, pm->part());

            if(pm->transformation().isIdentity())
            {
                this->centerPart(gop);
                this->dropPart(gop);

                PreferenceChoice should_shift = PM->getFileShiftPreference();
                if(should_shift == PreferenceChoice::kAsk)
                {
                    if (QMessageBox::question(this, "Warning", "Do you wish to shift "+ gop->part()->name() + " so that it does not intersect current parts?",
                                              QMessageBox::Yes|QMessageBox::No) == QMessageBox::Yes)
                        should_shift = PreferenceChoice::kPerformAutomatically;
                    else
                        should_shift = PreferenceChoice::kSkipAutomatically;
                }
                if(should_shift == PreferenceChoice::kPerformAutomatically)
                    this->shiftPart(gop);
            }
        }

        m_printer->adoptChild(gop);
        m_part_objects.insert(gop);

        pm->setGraphicsPart(gop);

        // Sub object visibility.
        gop->setOverhangAngle(m_sb->setting<Angle>(Constants::ProfileSettings::Support::kThresholdAngle));
        if (m_state.overhangs_shown) gop->showOverhang(true);
        if (m_state.planes_shown) gop->plane()->show();
        if (m_state.names_shown) gop->label()->show();

        this->blockModel();
        m_model->lookupByGraphic(gop)->setTranslation(gop->translation());
        this->permitModel();

        this->postTransformCheck();

        pm->setOriginalTransformation(gop->transformation());

        this->update();
    }

    void PartView::modelReloadUpdate(QSharedPointer<PartMetaItem> pm)
    {
        QSharedPointer<PartObject> gop = pm->graphicsPart();
        auto tfm = gop->transformation();

        modelRemovalUpdate(pm);
        pm->setGraphicsPart(nullptr);
        modelAdditionUpdate(pm);

        gop = pm->graphicsPart();
        pm->setTransformation(tfm);

        m_selected_objects.insert(gop);
        this->setCursor(QCursor(Qt::OpenHandCursor));
        gop->select();
        m_state.highlighted_part = gop;

        this->blockModel();
        m_model->lookupByGraphic(gop)->setSelected(true);
        this->permitModel();

        this->update();
    }

    void PartView::modelRemovalUpdate(QSharedPointer<PartMetaItem> pm)
    {
        QSharedPointer<PartObject> gop = pm->graphicsPart();

        m_part_objects.remove(gop);
        m_selected_objects.remove(gop);
        m_state.part_rot_start.remove(gop);
        m_state.part_trans_start.remove(gop);

        m_printer->orphanChild(gop);

        if (m_state.highlighted_part == gop) m_state.highlighted_part = nullptr;

        this->update();
    }

    void PartView::modelParentingUpdate(QSharedPointer<PartMetaItem> pm)
    {
        QSharedPointer<PartMetaItem> parent_pm = pm->parent();
        QSharedPointer<PartObject> parent_pm_gop = (parent_pm.isNull()) ? nullptr : parent_pm->graphicsPart();
        QSharedPointer<PartObject> gop = pm->graphicsPart();
        QSharedPointer<GraphicsObject> parent_gop = pm->graphicsPart()->parent();

        gop->unselect();
        if (parent_gop.dynamicCast<PartObject>()) parent_gop.dynamicCast<PartObject>()->unselect();
        if (parent_pm_gop != nullptr) parent_pm_gop->unselect();

        // Check parents - This is frankly confusing, so I'm going to step through it in good detail.

        // If the parent of the graphics part is not the same as the graphics part for this update, then
        // its parent has changed. It needs to be removed from the old parent and added to the new one.
        if (parent_pm_gop != parent_gop) {
            // Parent gop should always be valid, since if it isn't another part is should be the printer.
            parent_gop->orphanChild(gop);

            // If the new parent in the model is null, then the printer gets this object.
            if (!parent_pm_gop.isNull()) parent_pm_gop->adoptChild(gop);
            else m_printer->adoptChild(gop);
        }


        // Check if children need to be altered.
        for (auto& cpm : pm->children()) {
            // If the child has a parent but it is not the current part, it needs to be updated.
            if (cpm->graphicsPart()->parent() != gop) {
                // The parent is guarenteed to be valid since if it is not another part, it is the printer.
                // The object should be orphaned and then adopted by the new parent.
                cpm->graphicsPart()->parent()->orphanChild(cpm->graphicsPart());
                gop->adoptChild(cpm->graphicsPart());
            }
        }

        this->update();
    }

    void PartView::modelTranformUpdate(QSharedPointer<PartMetaItem> pm)
    {
        QSharedPointer<PartObject> gop = pm->graphicsPart();

        gop->setTransformation(pm->transformation());

        if (PM->getAlwaysDropParts() && !MathUtils::glEquals(gop->minimum().z(), m_printer->minimum().z()))
        {
            // Is this a z translation or not?
            if (MathUtils::glEquals(gop->translation().z() - pm->translation().z(), 0.0f))
            {
                dropPart(gop);
                this->blockModel();
                m_model->lookupByGraphic(gop)->setTranslation(gop->translation());
                this->permitModel();
            }
        }

        this->postTransformCheck();

        this->update();
    }

    void PartView::modelVisualUpdate(QSharedPointer<PartMetaItem> pm)
    {
        QSharedPointer<PartObject> gop = pm->graphicsPart();

        gop->setTransparency(pm->transparency());
        gop->setMeshTypeColor(pm->meshType());
        gop->setRenderMode(pm->renderMode());
        gop->setSolidWireFrameMode(pm->solidWireframeMode());
        this->update();
    }

    void PartView::postTransformCheck()
    {
        auto floating = this->floatingParts();
        auto external = this->externalParts();

        emit positioningIssues(external, floating);

        this->update();
    }

    void PartView::alignPart(QSharedPointer<PartObject> gop, Triangle tri, QVector3D plane_norm)
    {
        // From old graphics part.
        QVector3D floor_norm;
        QVector3D axis;
        float acos_arg;
        float acos_angle;
        float asin_arg;
        float asin_angle;
        float angle;
        QVector3D surface_norm;
        QVector3D local_a;
        QVector3D local_b;
        QVector3D local_c;
        QVector3D edge1;
        QVector3D edge2;
        QVector3D floor_center;

        floor_center = m_printer->printerCenter();

        //Get local triangle coordinates
        local_a = gop->transformation().inverted() * tri.a;
        local_b = gop->transformation().inverted() * tri.b;
        local_c = gop->transformation().inverted() * tri.c;

        //Find the triangle normal
        edge1 = local_b - local_a;
        edge2 = local_c - local_a;
        surface_norm = QVector3D::crossProduct(edge1, edge2);
        surface_norm.normalize();

        //Rotate the surface norm into the local coordinate frame
        floor_norm = gop->rotation().conjugated().rotatedVector(plane_norm);
        floor_norm.normalize();
        // From old graphics part.

        //Cross the two normals to find the rotation axis.
        axis = QVector3D::crossProduct(surface_norm, floor_norm);

        //Explicitly bound acos and asin by -1 and 1 to avoid NaN
        acos_arg = std::max( std::min(QVector3D::dotProduct(surface_norm, floor_norm)/ (surface_norm.length()*floor_norm.length()), 1.0f), -1.0f);
        asin_arg = std::max( std::min(axis.length() / (surface_norm.length()*floor_norm.length()), 1.0f), -1.0f);
        acos_angle = qAcos(acos_arg);
        asin_angle = qAsin(asin_arg);

        //Get correct value and sign of angle from what we know about
        //range of asin and acos
        if(acos_angle < M_PI_2 && asin_angle < 0.0f)
        {
            angle = asin_angle;
        }
        else if(acos_angle > M_PI_2 && asin_angle > 0.0f)
        {
            angle = acos_angle;
        }
        else if(acos_angle > M_PI_2 && asin_angle < 0.0f)
        {
            angle = -acos_angle;
        }
        else
        {
            angle = acos_angle;
        }

        //If axis is zero, this means the surface norm and floor norm are parallel or antiparallel
        if(qAbs(axis.x()) < 0.01 && qAbs(axis.y()) < 0.01 && qAbs(axis.z()) < 0.01)
        {
            //Parallel, so don't do rotate
            if(acos_arg > 0)
            {
                angle = 0.0f;
            }
            //Antiparallel, so axis doesn't really matter. Arbitrarily choose local x-axis
            else
            {
                axis = QVector3D(1,0,0);
            }
        }
        QQuaternion rotation = QQuaternion::fromAxisAndAngle(axis, qRadiansToDegrees(angle));
        gop->rotate(rotation);

        float z_sub = Constants::Limits::Maximums::kMaxFloat;

        // Find the actual minimum on the object.
        for (Triangle t : gop->triangles()) {
            for (uint i = 0; i < 3; i++) {
                QVector3D pt = t[i];

                if (pt.z() < z_sub) z_sub = pt.z();
            }
        }

        QVector3D trans = gop->translation();
        z_sub -= m_printer->printerCenter().z();
        trans.setZ(trans.z() - z_sub);

        gop->translateAbsolute(trans, true);

        this->blockModel();
        m_model->lookupByGraphic(gop)->setTranslation(gop->translation() - m_printer->minimum());
        m_model->lookupByGraphic(gop)->setRotation(gop->rotation());
        this->permitModel();
    }

    QSharedPointer<PartObject> PartView::pickPart(const QPointF& mouse_ndc_pos, QSet<QSharedPointer<PartObject>> object_set)
    {
        float min_dist = Constants::Limits::Maximums::kMaxFloat;

        QSharedPointer<PartObject> picked_part;

        for(auto& gop : object_set) {
            float dist = PartPicker::pickDistance(this->projectionMatrix(), this->viewMatrix(), mouse_ndc_pos, gop->triangles());

            if (dist < min_dist) {
                min_dist = dist;
                picked_part = gop;
            }
        }

        return picked_part;
    }

    QSharedPointer<PartObject> PartView::findObject(QString name)
    {
        // Find GOP
        QSharedPointer<PartObject> gop;
        for (auto& go : m_part_objects) {
            if (go->name() == name) {
                gop = go;
                break;
            }
        }

        return gop;
    }

    void PartView::blockModel()
    {
        QObject::disconnect(m_model.get(), &PartMetaModel::itemAddedUpdate,  this, &PartView::modelAdditionUpdate);
        QObject::disconnect(m_model.get(), &PartMetaModel::selectionUpdate,  this, &PartView::modelSelectionUpdate);
        QObject::disconnect(m_model.get(), &PartMetaModel::parentingUpdate,  this, &PartView::modelParentingUpdate);
        QObject::disconnect(m_model.get(), &PartMetaModel::transformUpdate,  this, &PartView::modelTranformUpdate);
        // Removal/visual update never needs to be disconnected as it is currently only modified by the right click menu/part widget (i.e. externally)
    }

    void PartView::permitModel()
    {
        QObject::connect(m_model.get(), &PartMetaModel::itemAddedUpdate,  this, &PartView::modelAdditionUpdate);
        QObject::connect(m_model.get(), &PartMetaModel::selectionUpdate,  this, &PartView::modelSelectionUpdate);
        QObject::connect(m_model.get(), &PartMetaModel::parentingUpdate,  this, &PartView::modelParentingUpdate);
        QObject::connect(m_model.get(), &PartMetaModel::transformUpdate,  this, &PartView::modelTranformUpdate);
    }
} // Namespace ORNL
