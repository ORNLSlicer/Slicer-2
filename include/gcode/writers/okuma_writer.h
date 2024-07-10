#ifndef OKUMA_WRITER_H
#define OKUMA_WRITER_H

//! \file okuma_writer.h

#include "gcode/writers/writer_base.h"
#include "managers/settings/settings_manager.h"
#include "gcode/gcode_meta.h"
namespace ORNL
{
    /*!
     * \class OkumaWriter
     * \brief The gcode writer for the Haas syntax
     */
    class OkumaWriter : public WriterBase
    {
    public:

        //! \brief Constructor
        OkumaWriter(GcodeMeta meta, const QSharedPointer<SettingsBase>& sb);

        //! \brief Writes initial setup instructions for the machine state
        QString writeInitialSetup(Distance minimum_x, Distance minimum_y, Distance maximum_x, Distance maximum_y, int num_layers) override;

         //! \brief Writes G-Code to be executed at the start of the layer
        QString writeBeforeLayer(float min_z, QSharedPointer<SettingsBase> sb) override;

         //! \brief Writes G-Code to be executed each layer before each part
        QString writeBeforePart(QVector3D normal) override;

        //! \brief Writes G-Code to be exectued before each island
        QString writeBeforeIsland() override;

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


        //! \brief Writes G-Code to be executed after each path
        QString writeAfterPath(RegionType type) override;

        //! \brief Writes G-Code to be executed after each region
        QString writeAfterRegion(RegionType type) override;

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

        //! \brief Writes G-Code to enable the extruder
        QString writeExtruderOn(RegionType type, int rpm);
        //! \brief Writes G-Code to disable the extruder
        QString writeExtruderOff();

        //! \brief Writes gcode coordinates WXYZ for a move or travel to the destination point
        QString writeCoordinates(Point destination);
        QString writeBrokenTravel(Point destination);

        AngularVelocity m_current_rpm;

        //! \brief true if first travel, false for subsequent travels
        bool m_first_travel;
        //! \brief true if first printing segment, false for subsquent paths - needed for Spiralize
        bool m_first_print;
        //! \brief true is first print motion of the layer
        bool m_layer_start;

        //! \brief store the previous line in m
        QString m_last_line;

        //! \brief preallocated prefixs commonly used in this syntax
        int m_material_number;

        //! \brief true means the last segment was a travel.
        int was_travel;

        //! \brief keep track of C axis position
        double c_angle;

        //! \brief keep track of inner vs outer perimeter
        bool m_inner_perimeter;
        bool m_outer_perimeter;

        //! \brief keep track of the layer
        bool m_new_layer;

    };  // class OkumaWriter
}  // namespace ORNL
#endif  // OKUMA_WRITER_H
