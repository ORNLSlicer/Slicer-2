#ifndef ORNL_WRITER_H
#define ORNL_WRITER_H

//! \file ornl_writer.h

#include "gcode/writers/writer_base.h"
#include "managers/settings/settings_manager.h"
#include "gcode/gcode_meta.h"

namespace ORNL
{
    /*!
     * \class ORNLWriter
     * \brief The gcode writer for the ORNL syntax
     */
    class ORNLWriter : public WriterBase
    {
    public:

        //! \brief Constructor
        ORNLWriter(GcodeMeta meta, const QSharedPointer<SettingsBase>& sb);

        //! \brief Writes important settings info to the header for the operator to read
        QString writeSettingsHeader(GcodeSyntax syntax);

        //! \brief Writes initial setup instructions for the machine state
        QString writeInitialSetup(Distance minimum_x, Distance minimum_y, Distance maximum_x, Distance maximum_y, int num_layers) override;

         //! \brief Writes G-Code to be executed at the start of the layer
        QString writeBeforeLayer(float min_z, QSharedPointer<SettingsBase> sb) override;

         //! \brief Writes G-Code to be executed each layer before each part
        QString writeBeforePart(QVector3D normal) override;

        //! \brief Writes G-Code to be exectued before each island
        QString writeBeforeIsland() override;

        //! \brief Writes G-Code to be executed before each scan
        QString writeBeforeScan(Point min, Point max, int layer, int boundingBox, Axis axis, Angle angle) override;

        //! \brief Writes G-Code to be executed at the start of each region
        QString writeBeforeRegion(RegionType type, int pathSize) override;

        //! \brief Writes G-Code to be executed at the start of each path
        QString writeBeforePath(RegionType type) override;

        //! \brief Writes G-Code for traveling between paths
        QString writeTravel(Point start_location, Point target_location, TravelLiftType lType,
                                            QSharedPointer<SettingsBase> params) override;

        //! \brief Writes G-Code for line
        QString writeLine(const Point& start_point,
                          const Point& target_point,
                          const QSharedPointer<SettingsBase> params) override;

        //! \brief Writes G-Code for arc
        QString writeArc(const Point &start_point,
                         const Point &end_point,
                         const Point &center_point,
                         const Angle &angle,
                         const bool &ccw,
                         const QSharedPointer<SettingsBase> params) override;

        //! \brief writes a spline using the G5 command
        //! \param start_point the starting location
        //! \param a_control_point first control point
        //! \param b_control_point second control point
        //! \param end_point the ending location
        //! \param params the settings base
        //! \return a string with the gcode command
        QString writeSpline(const Point& start_point,
                            const Point& a_control_point,
                            const Point& b_control_point,
                            const Point& end_point,
                            const QSharedPointer<SettingsBase> params) override;

        //! \brief Writes G-Code for scan
        QString writeScan(Point target_point, Velocity speed, bool on_off) override;

        //! \brief Writes G-Code to be executed after each path
        QString writeAfterPath(RegionType type) override;

        //! \brief Writes G-Code to be executed after each region
        QString writeAfterRegion(RegionType type) override;

        //! \brief Writes G-Code to be executed after each scan
        QString writeAfterScan(Distance beadWidth, Distance laserStep,
                               Distance laserResolution) override;

        //! \brief Writes G-Code to be executed at the end of each island
        QString writeAfterIsland() override;

        //! \brief Writes G-Code to be executed at the end of each part each layer
        QString writeAfterPart() override;

        //! \brief Writes G-Code to be executed at the end of each layer
        QString writeAfterLayer() override;

        //! \brief Writes G-Code for shutting down the machine
        QString writeShutdown() override;

        //! \brief Write purge command
        QString writePurge(int RPM, int duration, int delay) override;

        //! \brief Writes G-Code for a pause, G4
        QString writeDwell(Time time) override;

    private:

        //! \brief Writes G-Code to enable the tamper
        QString writeTamperOn();
        //! \brief Writes G-Code to disable the tamper
        QString writeTamperOff();
        //! \brief Writes G-Code to enable the extruder
        QString writeExtruderOn(RegionType type, int rpm, int extruder_number);
        //! \brief Writes G-Code to disable the extruder
        QString writeExtruderOff(int extruder_number);

        //! \brief Writes gcode coordinates WXYZ for a move or travel to the destination point
        QString writeCoordinates(Point destination);

        //! \brief State variables
        AngularVelocity m_current_rpm;
        int m_current_recipe;

        //! \brief true if first travel, false for subsequent travels
        bool m_first_travel;
        //! \brief true if first printing segment, false for subsquent paths - needed for Spiralize
        bool m_first_print;
        //! \brief true is first print motion of the layer
        bool m_layer_start;

    };  // class ORNLWriter
}  // namespace ORNL
#endif  // ORNL_WRITER_H
