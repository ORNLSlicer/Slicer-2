#pragma once

#include <QDialog>

namespace ORNL {
//! \brief Displays the changelog bundled with this build, including unreleased changes.
class ChangelogWindow : public QDialog {
   public:
    //! \brief Creates a read-only, nonmodal changelog window owned by parent.
    explicit ChangelogWindow(QWidget* parent = nullptr);
};
}  // namespace ORNL
