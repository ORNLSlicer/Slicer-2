#include "widgets/part_widget/part_widget.h"

// Qt
#include <QMessageBox>
#include <QFileDialog>

// Local
#include "graphics/view/part_view.h"
#include "managers/session_manager.h"
#include "managers/settings/settings_manager.h"
#include "utilities/mathutils.h"

namespace ORNL {
    PartWidget::PartWidget(QWidget *parent) : QWidget(parent) {
        this->setupWidget();
    }

    QSet<QSharedPointer<Part>> PartWidget::parts() {
        QSet<QSharedPointer<Part>> ret;

        // Update the parts with the manipulations from the view.
        for (auto& item : m_model->items()) {
            QSharedPointer<Part> p = item->part();

            QVector3D    gop_translation = item->translation();
            QQuaternion  gop_rotation    = item->rotation();
            QVector3D    gop_scale       = item->scale();

            gop_translation *= Constants::OpenGL::kViewToObject;

            QMatrix4x4 object_transformation = MathUtils::composeTransformMatrix(gop_translation, gop_rotation, gop_scale);

            // TODO (if it's important): update parent children here.

            p->setTransformation(object_transformation);
            p->rootMesh()->setType(item->meshType());

            ret.insert(p);
        }

        return ret;
    }

    QSharedPointer<PartMetaModel> PartWidget::getPartMeta()
    {
        return m_model;
    }

    QString PartWidget::getFirstPartName()
    {
        return m_part_control->nameOfFirstPart();
    }

    void PartWidget::takeScreenshot() {
        QString filepath;
        QImage screenshot;

        // TODO: maybe make this talk with CSM to save.
        filepath = QFileDialog::getSaveFileName(this, tr("Save Screenshot"), nullptr, tr("Images (*.png *.jpg *.gif *.tif)"));

        if (!filepath.isNull())
        {
            //If file has no file ending, just save as png
            if(!filepath.contains("."))
            {
                filepath += ".png";
            }
            screenshot = m_part_view->grabFramebuffer();
            screenshot.save(filepath);
        }
    }

    void PartWidget::undo() {

    }

    void PartWidget::redo() {

    }

    void PartWidget::copy() {
        m_model->setSelectionCopied();
    }

    void PartWidget::paste() {
        m_model->copySelection();
    }

    void PartWidget::add(QSharedPointer<Part> part) {
        auto pm = m_model->newItem(part);
        // Fixes a very bizzare bug where the selection update slot for the view is not
        // executed on only the first selection. After that, it appears to be fine. This
        // basically just forces a selection to occur first. From an hour of debugging,
        // it seems that the slot _is_ actually connect, the QObject just refuses to actually
        // send the signal. Maybe a bug in Qt? Or I'm just not seeing something?
        pm->setSelected(true);
        pm->setSelected(false);
    }

    void PartWidget::reload() {
        for (auto& pm : m_model->selectedItems()) {
            m_model->reloadItem(pm);
        }
    }

    void PartWidget::remove() {
        for (auto& pm : m_model->selectedItems()) {
            m_model->removeItem(pm);
        }
    }

    void PartWidget::remove(QSharedPointer<Part> part) {
        m_model->removeItem(m_model->lookupByPointer(part));
    }

    void PartWidget::clear() {
        m_model->clearItems();
    }

    void PartWidget::zoomIn() {
        m_part_view->zoomIn();
    }

    void PartWidget::zoomOut() {
        m_part_view->zoomOut();
    }

    void PartWidget::resetZoom() {
        m_part_view->resetZoom();
    }

    void PartWidget::resetCamera() {
        m_part_view->resetCamera();
    }

    void PartWidget::handleModifiedSetting(const QString& setting_key) {
        static const auto printer_settings = QSet<QString> {
            Constants::PrinterSettings::Dimensions::kXMin,
            Constants::PrinterSettings::Dimensions::kXMax,
            Constants::PrinterSettings::Dimensions::kYMin,
            Constants::PrinterSettings::Dimensions::kYMax,
            Constants::PrinterSettings::Dimensions::kZMin,
            Constants::PrinterSettings::Dimensions::kZMax,
            Constants::PrinterSettings::Dimensions::kXOffset,
            Constants::PrinterSettings::Dimensions::kYOffset,
            Constants::PrinterSettings::Dimensions::kWMin,
            Constants::PrinterSettings::Dimensions::kWMax,
            Constants::PrinterSettings::Dimensions::kBuildVolumeType,
            Constants::PrinterSettings::Dimensions::kInnerRadius,
            Constants::PrinterSettings::Dimensions::kOuterRadius,
            Constants::PrinterSettings::Dimensions::kEnableW,
            Constants::PrinterSettings::Dimensions::kEnableGridX,
            Constants::PrinterSettings::Dimensions::kGridXDistance,
            Constants::PrinterSettings::Dimensions::kGridXOffset,
            Constants::PrinterSettings::Dimensions::kEnableGridY,
            Constants::PrinterSettings::Dimensions::kGridYDistance,
            Constants::PrinterSettings::Dimensions::kGridYOffset
        };

        static const auto material_settings = QSet<QString> {
            Constants::ProfileSettings::SlicingAngle::kSlicingPlaneNormalX,
            Constants::ProfileSettings::SlicingAngle::kSlicingPlaneNormalY,
            Constants::ProfileSettings::SlicingAngle::kSlicingPlaneNormalZ
        };

        static const auto optimization_settings = QSet<QString> {
            Constants::ProfileSettings::Optimizations::kIslandOrder,
            Constants::ProfileSettings::Optimizations::kCustomIslandXLocation,
            Constants::ProfileSettings::Optimizations::kCustomIslandYLocation,
            Constants::ProfileSettings::Optimizations::kPathOrder,
            Constants::ProfileSettings::Optimizations::kCustomPathXLocation,
            Constants::ProfileSettings::Optimizations::kCustomPathYLocation,
            Constants::ProfileSettings::Optimizations::kCustomPointXLocation,
            Constants::ProfileSettings::Optimizations::kCustomPointYLocation,
            Constants::ProfileSettings::Optimizations::kEnableSecondCustomLocation,
            Constants::ProfileSettings::Optimizations::kCustomPointSecondXLocation,
            Constants::ProfileSettings::Optimizations::kCustomPointSecondYLocation,
            Constants::PrinterSettings::Dimensions::kXOffset,
            Constants::PrinterSettings::Dimensions::kYOffset
        };

        static const auto overhang_settings = QSet<QString> {
            Constants::ProfileSettings::Support::kThresholdAngle
        };

        if(printer_settings.contains(setting_key)) {
            m_part_view->updatePrinterSettings(GSM->getGlobal());
        }
        else if (material_settings.contains(setting_key)) {
            m_part_view->updateSlicingSettings(GSM->getGlobal());
        }
        else if(optimization_settings.contains(setting_key)) {
            m_part_view->updateOptimizationSettings(GSM->getGlobal());
        }
        else if(overhang_settings.contains(setting_key)) {
            m_part_view->updateOverhangSettings(GSM->getGlobal());
        }
    }

    void PartWidget::setEnabled(bool status) {
        m_toolbar->setEnabled(status);
    }

    void PartWidget::preSliceUpdate() {
        CSM->clearParts();

        auto parts = this->parts();

        for (auto p : parts) {
            CSM->addPart(p, false);
        }

        emit slice();
    }

    void PartWidget::showSlicingPlanes(bool show)
    {
        m_part_view->showSlicingPlanes(show);
    }

    void PartWidget::showLabels(bool show)
    {
        m_part_view->showLabels(show);
    }

    void PartWidget::showSeams(bool show)
    {
        m_part_view->showSeams(show);
    }

    void PartWidget::showOverhang(bool show)
    {
        m_part_view->showOverhang(show);
    }

    void PartWidget::updatePartTransformations()
    {
        for (auto& item : m_model->items())
        {
            QSharedPointer<Part> p = item->part();

            QVector3D    gop_translation = item->translation();
            QQuaternion  gop_rotation    = item->rotation();
            QVector3D    gop_scale       = item->scale();

            gop_translation *= Constants::OpenGL::kViewToObject;

            QMatrix4x4 object_transformation = MathUtils::composeTransformMatrix(gop_translation, gop_rotation, gop_scale);

            // TODO (if it's important): update parent children here.

            p->setTransformation(object_transformation);
            p->rootMesh()->setType(item->meshType());
        }
    }

    void PartWidget::modelAdditionUpdate(QSharedPointer<PartMetaItem> item) {
        emit added(item->part());
    }

    void PartWidget::modelSelectionUpdate(QSharedPointer<PartMetaItem> item) {
        QList<QSharedPointer<PartMetaItem>> selected_items = m_model->selectedItems();
        QSet<QSharedPointer<Part>> selected_set;

        QSharedPointer<PartMetaItem> manip_item;

        if (item->isSelected()) manip_item = item;
        else if (selected_items.count()) manip_item = selected_items.back();

        if (manip_item.isNull()) this->setStatusSelection("None");
        else this->setStatusSelection(manip_item->part()->name());

        for (auto& pm : selected_items) {
            selected_set.insert(pm->part());
        }

        this->setEnabled(!selected_set.empty());

        emit selected(selected_set, (manip_item.isNull()) ? nullptr : manip_item->part());
    }

    void PartWidget::modelTransformationUpdate(QSharedPointer<PartMetaItem> item) {
        if(item->rotation() != QQuaternion(1,0,0,0) && item->scale() != QVector3D(1,1,1))
            emit displayRotationInfoMsg();

        // IMPORTANT NOTE: there is no intention of keeping this as is. The layer bar should
        // work off of the PartMetaItem, not the part itself. The this currently exists just to
        // keep the bar in working order and should be replaced as soon as the layerbar is either
        // updated or removed.

        // ANOTHER IMPORTANT NOTE: The connection to the layerbar from this function is culprit for the slowdown
        // while translating. This is yet another reason for a layerbar rework.

        // A THIRD IMPORTANT NOTE: This slot is disconnected right now.

        /*
        QSharedPointer<Part> p = item->part();

        QVector3D    gop_translation = item->translation();
        QQuaternion  gop_rotation    = item->rotation();
        QVector3D    gop_scale       = item->scale();

        gop_translation *= Constants::OpenGL::kViewToObject;

        QMatrix4x4 object_transformation = MathUtils::composeTransformMatrix(gop_translation, gop_rotation, gop_scale);

        p->setTransformation(object_transformation);

        emit modified(p);
        */
    }

    void PartWidget::modelRemovalUpdate(QSharedPointer<PartMetaItem> item) {
        this->setStatusSelection("None");
        this->positionIssues(QList<QSharedPointer<Part>>() , QList<QSharedPointer<Part>>());

        this->setEnabled(false);

        CSM->removePart(item->part());

        emit removed(item->part());
        emit selected(QSet<QSharedPointer<Part>>(), nullptr);
    }

    void PartWidget::positionIssues(QList<QSharedPointer<Part>> opl, QList<QSharedPointer<Part>> fpl)
    {
        for(auto part_meta_item : m_model->items()) // Clear old status
        {
            m_part_control->setFloatingStatus(part_meta_item->part()->name(), false);
            m_part_control->setOutsideStatus(part_meta_item->part()->name(), false);
        }

        for (auto& part : opl)
            m_part_control->setOutsideStatus(part->name(), true);

        for (auto& part : fpl)
            m_part_control->setFloatingStatus(part->name(), true);
    }

    void PartWidget::setStatusSelection(QString name) {
        m_status_state.selected_part = name;

        m_accentColor = PreferencesManager::getInstance()->getTheme().getDotPairedColor().name();
        QString status = "Currently Manipulating: <font color=\"" % m_accentColor % "\">%1</font>";
        status = status.arg(m_status_state.selected_part);

        m_selection_label->setText(status);
        m_selection_label->setMinimumSize(200, 17); //size when no part is loaded
        m_selection_label->adjustSize();
    }

    void PartWidget::setStatusIssue(QString issue) {
        m_status_state.issues = issue;
    }

    void PartWidget::resizeEvent(QResizeEvent* event) {
        QPoint new_size = QPoint(event->size().width(), event->size().height());

        m_part_control->move(Constants::UI::PartControl::kLeftOffset, event->size().height() - (Constants::UI::PartControl::kSize.height() + Constants::UI::PartControl::kBottomOffset));
        m_selection_label->move(Constants::UI::PartControl::kLeftOffset + 10, event->size().height() - (Constants::UI::PartControl::kSize.height() + 10 + m_selection_label->height()));

        emit resized(event->size());
    }

    void PartWidget::setupWidget()
    {
        m_model = QSharedPointer<PartMetaModel>::create();

        this->setupSubWidgets();
        this->setupLayouts();
        this->setupPosition();
        this->setupInputs();
        this->setupEvents();
    }

    void PartWidget::setupSubWidgets()
    {
        // Set OpenGL Version information
        // Note: This format must be set before show() is called.
        QSurfaceFormat format;
        format.setRenderableType(QSurfaceFormat::OpenGL);
        format.setProfile(QSurfaceFormat::CoreProfile);
        format.setSamples(4);
        format.setVersion(3, 3);

        // OpenGL View
        m_part_view = new PartView(GSM->getGlobal());
        m_part_view->resize(this->width(), this->height());
        m_part_view->setFormat(format);
        m_part_view->setModel(m_model);

        // Toolbar
        m_toolbar = new PartToolbar(m_model, this);
        m_toolbar->raise();

        // View Controls
        m_view_controls = new ViewControlsToolbar(this);
        m_view_controls->raise();

        // Selection
        m_selection_label = new QLabel(this);
        this->setStatusSelection("None");
        m_selection_label->show();

        // Part Control
        m_part_control = new PartControl(this);
        m_part_control->resize(Constants::UI::PartControl::kSize);
        m_part_control->show();
        m_part_control->setModel(m_model);
    }

    void PartWidget::setupStyle()
    {
        m_toolbar->setupStyle();
        m_view_controls->setupStyle();
        m_part_view->setupStyle();
        m_part_control->setupStyle();
        m_accentColor = PreferencesManager::getInstance()->getTheme().getDotPairedColor().name();
        QString status = "Currently Manipulating: <font color=\"" % m_accentColor % "\">%1</font>";
        status = status.arg(m_status_state.selected_part);
        m_selection_label->setText(status);
    }

    void PartWidget::setupLayouts()
    {
        m_layout = new QVBoxLayout(this);
        m_layout->addWidget(m_part_view);
    }

    void PartWidget::setupPosition()
    {
        // Status
        m_selection_label->move(300, this->height() - (Constants::UI::PartControl::kSize.height() + 15 + m_selection_label->height()));

        // Projection Buttons
        QPoint new_size = QPoint(this->width(), this->height());

        // Part control
        m_part_control->move(Constants::UI::PartControl::kLeftOffset, this->height() - (Constants::UI::PartControl::kSize.height() + Constants::UI::PartControl::kBottomOffset));
    }

    void PartWidget::setupInputs()
    {
        m_part_view->lower();

        // Turn off all inputs by default until a part is selected
        this->setEnabled(false);
    }

    void PartWidget::setupEvents() {
        // Model -> This : Connect changes in model to updates for overall widget.
        connect(m_model.get(), &PartMetaModel::selectionUpdate, this, &PartWidget::modelSelectionUpdate);
        connect(m_model.get(), &PartMetaModel::itemAddedUpdate, this, &PartWidget::modelAdditionUpdate);
        connect(m_model.get(), &PartMetaModel::itemRemovedUpdate, this, &PartWidget::modelRemovalUpdate);
        connect(m_model.get(), &PartMetaModel::transformUpdate, this, &PartWidget::modelTransformationUpdate);

        // PrefManager -> PartWidget : Connect unit updates with the update functions in this class.
        connect(PM.get(), &PreferencesManager::distanceUnitChanged, this, [this](Distance new_unit, Distance old_unit){ m_toolbar->updateTranslationUnits(new_unit, old_unit);});
        connect(PM.get(), &PreferencesManager::angleUnitChanged, this, [this](Angle new_unit, Angle old_unit){ m_toolbar->updateAngleUnits(new_unit, old_unit);});

        connect(m_toolbar, &PartToolbar::centerParts, [this]() { m_part_view->centerPart(m_status_state.selected_part); });
        connect(m_toolbar, &PartToolbar::dropPartsToFloor, m_part_view, &PartView::dropSelectedParts);
        connect(this, &PartWidget::resized, m_toolbar, &PartToolbar::resize);
        connect(this, &PartWidget::resized, m_view_controls, &ViewControlsToolbar::resize);

        connect(m_toolbar, &PartToolbar::setupAlignment, m_part_view, &PartView::setupAlignment);

        // Buttons -> PartWidget: Connect view buttons
        connect(m_view_controls, &ViewControlsToolbar::setIsoView, m_part_view, &PartView::setForwardView);
        connect(m_view_controls, &ViewControlsToolbar::setFrontView, m_part_view, &PartView::setFrontView);
        connect(m_view_controls, &ViewControlsToolbar::setSideView, m_part_view, &PartView::setSideView);
        connect(m_view_controls, &ViewControlsToolbar::setTopView, m_part_view, &PartView::setTopView);

        // View -> This : Status updates about state.
        connect(m_part_view, &PartView::positioningIssues, this, &PartWidget::positionIssues);
    }

} // Namespace ORNL
