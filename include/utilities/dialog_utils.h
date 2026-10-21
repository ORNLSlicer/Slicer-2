#pragma once

#include <QLineEdit>
#include <QSharedPointer>
#include <QString>

class QWidget;

namespace ORNL {
class Part;

namespace DialogUtils {

//! \brief Positions and centers a dialog or secondary window on the same monitor (QScreen)
//!        as its parent window (or the currently active application window).
//! \param widget The dialog or top-level window to position.
//! \param parent The parent widget whose top-level window and screen should be matched.
void positionOnParentScreen(QWidget* widget, QWidget* parent = nullptr);

//! \brief Opens a modal text input dialog positioned on the parent window's monitor.
//! \param parent The parent widget for the modal input dialog.
//! \param title The window title for the dialog.
//! \param label The label text shown inside the dialog.
//! \param mode The echo mode for the line edit.
//! \param text The initial pre-filled text.
//! \param ok Optional pointer to a bool set to true if accepted, false if canceled.
//! \return The text entered by the user, or empty if canceled.
QString getText(QWidget* parent, const QString& title, const QString& label,
                QLineEdit::EchoMode mode = QLineEdit::Normal, const QString& text = "", bool* ok = nullptr);

//! \brief Prompts the user for a part or mesh name with duplicate validation.
//!        Re-prompts if the entered name is already used by a different part.
//! \param parent The parent widget for the modal input dialog.
//! \param title The window title for the dialog.
//! \param current_name The initial text pre-filled in the dialog (used when renaming to allow keeping the same name).
//! \param current_part The part being renamed (optional), so keeping its own name is not treated as a collision.
//! \return The name entered by the user, or empty if the user canceled.
QString promptForName(QWidget* parent, const QString& title, const QString& current_name = "",
                      QSharedPointer<Part> current_part = nullptr);

//! \brief Opens a dialog that asks for a number to be entered with units.
//! \param parent The parent widget for the modal input dialog.
//! \param label_text The label for the dialog.
//! \param unit_text The text to display for the unit.
//! \param unit_conversion The conversion value to apply to the input.
//! \param ok Flag that is set to true on success, false if the user canceled.
//! \return The converted value (0.0 if canceled).
double promptForSize(QWidget* parent, const QString& label_text, const QString& unit_text, double unit_conversion,
                     bool& ok);

}  // namespace DialogUtils
}  // namespace ORNL
