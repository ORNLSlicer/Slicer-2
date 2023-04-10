#include "windows/dialogs/emboss_dialog.h"

// Qt
#include <QGridLayout>
#include <QHBoxLayout>
#include <QSizePolicy>
#include <QToolButton>
#include <QLabel>
#include <QComboBox>
#include <QTableWidget>
#include <QHeaderView>
#include <QLineEdit>
#include <QRadioButton>
#include <QDoubleSpinBox>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>
#include <QListWidget>

// Local
#include "graphics/view/emboss_view.h"
#include "graphics/view/preview_view.h"
#include "geometry/mesh/closed_mesh.h"
#include "geometry/mesh/mesh_factory.h"
#include "managers/session_manager.h"
#include "utilities/debugutils.h"


namespace ORNL {
    EmbossDialog::EmbossDialog(QSet<QSharedPointer<Part>> session_set, QWidget* parent) : QDialog(parent) {
        this->setupUi();

        m_session_parts = session_set;
        m_load_radio->click();
    }

    QSharedPointer<Part> EmbossDialog::resultPart() {
        return m_preview->getBase();
    }


    void EmbossDialog::baseObjectFromFile() {
        QFileDialog load_dialog;
        load_dialog.setWindowTitle("Open Base Object Model");
        load_dialog.setAcceptMode(QFileDialog::AcceptOpen);
        load_dialog.setFileMode(QFileDialog::ExistingFile);
        load_dialog.setNameFilters(QStringList() << "STL Model File (*.stl)" << "Any Files (*)");
        load_dialog.setDirectory(CSM->getMostRecentModelLocation());

        if (!load_dialog.exec()) return;

        QString file = load_dialog.selectedFiles()[0];

        QString name = file.split('/').back();
        name = name.left(name.lastIndexOf('.'));

        // Set mesh name
        QString org_name = name;
        uint count = 1;

        // Try to find a name for this mesh.
        while (CSM->parts().contains(name)) {
            name = org_name + "_" + QString::number(count);
            count++;
        }

        // TODO: Temp usage of blocking call here. Go back and use LoadComplete() callback with mesh loader.
        auto p = DebugUtils::getPartFromFile(file, name, MeshType::kBuild);

        if (p.isNull()) {
            QMessageBox::critical(this, "Unable to load '" + file + "'",
                                  "File '" + file + "' was not correctly parsed.");
            return;
        }


        m_base_part = p;
        m_preview->setBasePart(p);

        m_file_line->setText(file);

        m_base_scale->setValue(QVector3D(1.0f, 1.0f, 1.0f));
        m_base_scale->setDisabled(false);
        m_add_button->setDisabled(false);
    }

    void EmbossDialog::baseObjectLoadCompleted(QSharedPointer<Part> p) {

    }

    void EmbossDialog::baseObjectFromSession(QString name) {
        auto p = CSM->getPart(name);

        auto new_part = QSharedPointer<Part>::create(p);
        new_part->setName(name + "_emboss");
        new_part->setTransformation(QMatrix4x4());

        m_preview->setBasePart(new_part);

        m_base_scale->setValue(QVector3D(1.0f, 1.0f, 1.0f));
        m_base_scale->setDisabled(false);
        m_add_button->setDisabled(false);
    }

    void EmbossDialog::baseObjectFromGenerator() {
        auto mesh = QSharedPointer<ClosedMesh>::create(MeshFactory::CreateBoxMesh(0.25 * m, 0.05 * m, 0.25 * m));
        mesh->setType(MeshType::kBuild);

        // Set mesh name
        QString name = "_internal_gen_prisim_mesh";
        QString org_name = name;
        uint count = 1;

        // Try to find a name for this mesh.
        while (CSM->parts().contains(name)) {
            name = org_name + "_" + QString::number(count);
            count++;
        }
        mesh->setName(name);

        auto p = QSharedPointer<Part>::create(mesh);

        m_base_part = p;
        m_preview->setBasePart(p);

        m_base_scale->setValue(QVector3D(1.0f, 1.0f, 1.0f));
        m_base_scale->setDisabled(false);
        m_add_button->setDisabled(false);
    }

    void EmbossDialog::embossObjectFromFile() {
        QFileDialog load_dialog;
        load_dialog.setWindowTitle("Open Emboss Object Model(s)");
        load_dialog.setAcceptMode(QFileDialog::AcceptOpen);
        load_dialog.setFileMode(QFileDialog::ExistingFiles);
        load_dialog.setNameFilters(QStringList() << "STL Model File (*.stl)" << "Any Files (*)");
        load_dialog.setDirectory(CSM->getMostRecentModelLocation());

        if (!load_dialog.exec()) return;

        m_emboss_parts.clear();

        for (QString file : load_dialog.selectedFiles()) {
            QString name = file.split('/').back();
            name = name.left(name.lastIndexOf('.'));

            // Set mesh name
            QString org_name = name;
            uint count = 1;

            // Try to find a name for this mesh.
            while (CSM->parts().contains(name)) {
                name = org_name + "_" + QString::number(count);
                count++;
            }

            while (m_scales.contains(name)) {
                name = org_name + "_" + QString::number(count);
                count++;
            }

            // TODO: Temp usage of blocking call here. Go back and use LoadComplete() callback with mesh loader.
            QSharedPointer<Part> p = DebugUtils::getPartFromFile(file, name, MeshType::kEmbossSubmesh);

            if (p.isNull()) {
                QMessageBox::critical(this, "Unable to load '" + file + "'",
                                      "File '" + file + "' was not correctly parsed.");
                return;
            }

            m_emboss_parts.push_back(p);

            m_part_list->addItem(p->name());
            m_preview->addEmbossPart(p);

            m_scales[p->name()] = QVector3D(1.0f, 1.0f, 1.0f);
        }

        m_part_scale->setValue(QVector3D(1.0f, 1.0f, 1.0f));
    }

    void EmbossDialog::embossObjectLoadCompleted() {

    }

    void EmbossDialog::accept() {
        this->QDialog::accept();
    }

    void EmbossDialog::reject() {
        m_preview->clear();
        this->QDialog::reject();
    }

    void EmbossDialog::selectPart(QString name) {
        m_selected_name = name;

        if (name.isEmpty()) {
            m_part_scale->setEnabled(false);
            m_part_list->clearSelection();
            m_part_list->setCurrentItem(nullptr);
            return;
        }

        auto pl = m_part_list->findItems(name, Qt::MatchExactly);
        m_part_list->setItemSelected(pl.at(0), true);

        m_part_scale->setEnabled(true);
        m_preview->selectPart(name);
        m_part_scale->setValue(m_scales[m_selected_name]);
    }

    void EmbossDialog::clear() {
        m_part_scale->setEnabled(false);

        m_part_scale->setValue(QVector3D(1.0f, 1.0f, 1.0f));
        m_base_scale->setValue(QVector3D(1.0f, 1.0f, 1.0f));

        m_base_part.reset();
        m_emboss_parts.clear();

        m_preview->clear();

        m_part_list->clear();
    }

    /*
    void EmbossDialog::loadModel() {
        QSharedPointer<Part> p = DebugUtils::getPartFromFile("/home/dd/slicer2/misc/ornl_leaf.stl", "foobar", MeshType::kBuild);

        m_preview->setPart(p);
    }

    void EmbossDialog::showPlane() {
        static bool toggle = false;
        toggle = !toggle;

        m_preview->showSlicingPlane(toggle);

    }

    void EmbossDialog::updateAxis(int idx) {
        // Set the transform by switching on an int and selecting the corresponding axis.
        // This relies on the fact that the combo box goes through the planes in this order: YZ, XZ, XY;
        switch(idx) {
            case 0: { // YZ
                m_preview->setSlicingPlaneRotation(0, Angle().from(90, deg), 0);
                break;
            }
            case 1: { // XZ
                m_preview->setSlicingPlaneRotation(Angle().from(90, deg), 0, 0);
                break;
            }
            case 2: { // XY
                m_preview->setSlicingPlaneRotation(0, 0, 0);
                break;
            }
        }
    }
    */

    void EmbossDialog::setupUi() {
        this->setupWindow();
        this->setupWidgets();
        this->setupLayouts();
        this->setupInsert();
        this->setupEvents();
    }


    void EmbossDialog::setupWindow() {
        this->setWindowTitle("Emboss Import");
        this->setMinimumSize(1280, 720);
    }

    void EmbossDialog::setupWidgets() {
        // Labels
        m_build_label = new QLabel(this);
        m_build_label->setText("Build Setup");
        m_build_label->setStyleSheet("border-bottom-width: 1px; border-bottom-style: solid; border-radius: 0px;"
                                     "font: 15px; font-weight: bold;");
        m_emboss_label = new QLabel(this);
        m_emboss_label->setText("Embossing Objects:");
        m_base_label = new QLabel(this);
        m_base_label->setText("Base Object:");
        m_preview_label = new QLabel(this);
        m_preview_label->setText("Preview:");
        //m_rule_label = new QLabel(this);
        //m_rule_label->setText("");
        //m_rule_label->setStyleSheet("border-bottom-width: 1px; border-bottom-style: solid; border-radius: 0px;");

        // Preview
        QSurfaceFormat format;
        format.setRenderableType(QSurfaceFormat::OpenGL);
        format.setProfile(QSurfaceFormat::CoreProfile);
        format.setSamples(4);
        format.setVersion(3, 3);

        m_preview = new EmbossView;
        m_preview->setFormat(format);
        m_preview->setMinimumSize(600, 600);

        // Table
        m_part_list = new QListWidget(this);
        /*
        m_part_list->setHorizontalHeaderLabels(QStringList() << "Object" << "Mode" << "Orientation");
        m_part_list->horizontalHeader()->setStretchLastSection(true);
        m_part_list->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
        */
        m_part_list->setMinimumWidth(460);
        m_part_list->setSelectionMode(QAbstractItemView::SingleSelection);

        // Buttons
        m_add_button = new QToolButton(this);
        m_add_button->setText("+");
        m_add_button->setToolTip("Add parts to emboss (a base object must be selected first)");
        m_add_button->setDisabled(true);

        m_browse_button = new QToolButton(this);
        m_browse_button->setText("...");

        // File Select
        m_file_line = new QLineEdit(this);
        m_file_line->setPlaceholderText("Select a file to load...");
        m_file_line->setDisabled(true);
        m_file_line->setReadOnly(true);

        // Radios
        m_load_radio = new QRadioButton(this);
        m_session_radio = new QRadioButton(this);
        m_generate_radio = new QRadioButton(this);

        m_load_radio->setText("Load new object");
        m_session_radio->setText("Select object from current session");
        m_generate_radio->setText("Generate wall");

        // Part list
        m_session_list = new QListWidget(this);
        for (auto& p : CSM->parts()) {
            QString name = p->name();
            m_session_list->addItem(name);
        }

        if (m_session_list->count() == 0) {
            m_session_list->addItem("No parts in session.");
            m_session_list->setDisabled(true);
        }

        // Base scaling
        m_base_scale = new XYZInputWidget(this);
        m_base_scale->showLock(true);
        m_base_scale->setLock(true);
        m_base_scale->showLabel(true);
        m_base_scale->setLabelText("Scale:");
        m_base_scale->setValue(QVector3D(1.0f, 1.0f, 1.0f));
        m_base_scale->setIncrement(0.1f);

        m_base_scale->setDisabled(true);

        // Part scaling
        m_part_scale = new XYZInputWidget(this);
        m_part_scale->showLock(true);
        m_part_scale->setLock(true);
        m_part_scale->showLabel(true);
        m_part_scale->setLabelText("Scale:");
        m_part_scale->setValue(QVector3D(1.0f, 1.0f, 1.0f));
        m_part_scale->setIncrement(0.1f);

        m_part_scale->setDisabled(true);

        // Button container
        m_button_container = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    }

    void EmbossDialog::setupLayouts() {
        m_layout = new QGridLayout(this);

        // m_load_layout becomes child of m_layout when added below.
        m_load_layout = new QGridLayout();
        m_load_layout->setColumnStretch(1, 1);
    }

    void EmbossDialog::setupInsert() {
        m_layout->addWidget(m_build_label, 0, 0, 1, 3);
        m_layout->setRowMinimumHeight(0, 50);
        m_layout->addWidget(m_emboss_label, 1, 0);
        m_layout->addWidget(m_add_button, 1, 1, Qt::AlignRight);
        m_layout->addWidget(m_part_list, 2, 0, 1, 2);

        m_layout->addWidget(m_preview_label, 1, 2);
        m_layout->addWidget(m_preview, 2, 2, 2, 1);

        m_layout->addWidget(m_part_scale, 3, 0, 1, 2);

        m_load_layout->addWidget(m_base_label, 0, 0, 1, 2);
        m_load_layout->addWidget(m_load_radio, 1, 0, 1, 3);
        m_load_layout->addWidget(m_file_line, 2, 0, 1, 2);
        m_load_layout->addWidget(m_browse_button, 2, 3);
        m_load_layout->addWidget(m_session_radio, 3, 0, 1, 3);
        m_load_layout->addWidget(m_session_list, 4, 0, 1, 3);
        m_load_layout->addWidget(m_generate_radio, 5, 0, 1, 3);

        m_layout->addLayout(m_load_layout, 4, 0, 1, 2);
        m_layout->addWidget(m_base_scale, 5, 0, 1, 2);

        m_layout->addWidget(m_button_container, 6, 2);
    }

    void EmbossDialog::setupEvents() {
        connect(m_load_radio, &QRadioButton::clicked, m_file_line, &QLineEdit::show);
        connect(m_load_radio, &QRadioButton::clicked, m_file_line, &QLineEdit::clear);
        connect(m_load_radio, &QRadioButton::clicked, m_browse_button, &QToolButton::show);
        connect(m_load_radio, &QRadioButton::clicked, this, &EmbossDialog::clear);
        connect(m_load_radio, &QRadioButton::clicked, m_session_list, &QListWidget::hide);
        connect(m_load_radio, &QRadioButton::clicked, this,
            [this]() {
                m_base_scale->setDisabled(true);
                m_add_button->setDisabled(true);
                m_scales.clear();
            }
        );

        connect(m_session_radio, &QRadioButton::clicked, m_session_list, &QListWidget::show);
        connect(m_session_radio, &QRadioButton::clicked, m_session_list, &QListWidget::reset);
        connect(m_session_radio, &QRadioButton::clicked, m_file_line, &QLineEdit::hide);
        connect(m_session_radio, &QRadioButton::clicked, this, &EmbossDialog::clear);
        connect(m_session_radio, &QRadioButton::clicked, m_browse_button, &QToolButton::hide);
        connect(m_session_radio, &QRadioButton::clicked, this,
            [this]() {
                m_base_scale->setDisabled(true);
                m_add_button->setDisabled(true);
                m_scales.clear();
            }
        );

        connect(m_generate_radio, &QRadioButton::clicked, m_file_line, &QLineEdit::hide);
        connect(m_generate_radio, &QRadioButton::clicked, m_browse_button, &QToolButton::hide);
        connect(m_generate_radio, &QRadioButton::clicked, m_session_list, &QListWidget::hide);

        connect(m_browse_button, &QToolButton::clicked, this, &EmbossDialog::baseObjectFromFile);
        connect(m_session_list, &QListWidget::currentTextChanged, this, &EmbossDialog::baseObjectFromSession);
        connect(m_generate_radio, &QRadioButton::clicked, this, &EmbossDialog::baseObjectFromGenerator);

        connect(m_add_button, &QToolButton::clicked, this, &EmbossDialog::embossObjectFromFile);

        connect(m_base_scale, QOverload<QVector3D>::of(&XYZInputWidget::valueChanged), m_preview, &EmbossView::scaleBasePart);
        connect(m_base_scale, QOverload<QVector3D>::of(&XYZInputWidget::valueChanged), this,
            [this](QVector3D s) {
                auto r = m_preview->scaleBasePart(s);

                for (auto& scaling : r) {
                    m_scales[scaling.first] = scaling.second;
                    if (m_selected_name == scaling.first) {
                        m_part_scale->setValue(scaling.second);
                    }
                }
            }
        );
        connect(m_part_scale, QOverload<QVector3D>::of(&XYZInputWidget::valueChanged), this,
            [this](QVector3D s) {
                m_scales[m_selected_name] = s;
                m_preview->scaleSelectedPart(s);
            }
        );

        connect(m_part_list, &QListWidget::currentTextChanged, this, &EmbossDialog::selectPart);
        connect(m_preview, &EmbossView::selected, this, &EmbossDialog::selectPart);

        QPushButton* ok = m_button_container->button(QDialogButtonBox::Ok);
        QPushButton* cancel = m_button_container->button(QDialogButtonBox::Cancel);
        connect(ok, &QPushButton::pressed, this, &EmbossDialog::accept);
        connect(cancel, &QPushButton::pressed, this, &EmbossDialog::reject);
    }

} // namespace ORNL
