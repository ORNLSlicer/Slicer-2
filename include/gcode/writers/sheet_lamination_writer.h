#ifndef SHEET_LAMINATION_WRITER_H
#define SHEET_LAMINATION_WRITER_H

#include "managers/settings/settings_manager.h"

#include "gcode/writers/writer_base.h"

#include "gcode/gcode_meta.h"

namespace ORNL
{
    class SheetLaminationWriter : public WriterBase
    {
    public:
        SheetLaminationWriter(GcodeMeta meta, const QSharedPointer<SettingsBase>& sb);

        // WriterBase interface
    public:
        //! \brief writeSlicerHeader
        //! \param syntax
        //! \return DFX file header in "999" comment form
        QString writeSlicerHeader(const QString& syntax) override;

        //! \brief writeSettingsHeader - Implemented to satisfy writer base interface. Unused by this syntax.
        //! \param syntax
        //! \return
        QString writeSettingsHeader(GcodeSyntax syntax) override;

        //! \brief writeInitialSetup - Implemented to satisfy writer base interface. Unused by this syntax.
        //! \param minimum_x
        //! \param minimum_y
        //! \param maximum_x
        //! \param maximum_y
        //! \param num_layers
        //! \return
        QString writeInitialSetup(Distance minimum_x, Distance minimum_y, Distance maximum_x, Distance maximum_y, int num_layers) override;

        //! \brief writeBeforeLayer - Implemented to satisfy writer base interface. Unused by this syntax.
        //! \param min_z
        //! \param sb
        //! \return
        QString writeBeforeLayer(float min_z, QSharedPointer<SettingsBase> sb) override;

        //! \brief writeBeforePart - Implemented to satisfy writer base interface. Unused by this syntax.
        //! \param normal
        //! \return
        QString writeBeforePart(QVector3D normal) override;

        //! \brief writeBeforeIsland - Implemented to satisfy writer base interface. Unused by this syntax.
        //! \return
        QString writeBeforeIsland() override;

        //! \brief writeBeforeRegion - Implemented to satisfy writer base interface. Unused by this syntax.
        //! \param type
        //! \param pathSize
        //! \return
        QString writeBeforeRegion(RegionType type, int pathSize) override;

        //! \brief writeBeforePath - Implemented to satisfy writer base interface. Unused by this syntax.
        //! \param type
        //! \return
        QString writeBeforePath(RegionType type) override;

        //! \brief writeTravel - Implemented to satisfy writer base interface. Unused by this syntax.
        //! \param start_location
        //! \param target_location
        //! \param lType
        //! \param params
        //! \return
        QString writeTravel(Point start_location, Point target_location, TravelLiftType lType, QSharedPointer<SettingsBase> params) override;

        //! \brief writeLine - Implemented to satisfy writer base interface. Unused by this syntax.
        //! \param start_point
        //! \param target_point
        //! \param params
        //! \return
        QString writeLine(const Point &start_point, const Point &target_point, const QSharedPointer<SettingsBase> params) override;

        //! \brief writeAfterPath - Implemented to satisfy writer base interface. Unused by this syntax.
        //! \param type
        //! \return
        QString writeAfterPath(RegionType type) override;

        //! \brief writeAfterRegion - Implemented to satisfy writer base interface. Unused by this syntax.
        //! \param type
        //! \return
        QString writeAfterRegion(RegionType type) override;

        //! \brief writeAfterIsland - Implemented to satisfy writer base interface. Unused by this syntax.
        //! \return
        QString writeAfterIsland() override;

        //! \brief writeAfterPart - Implemented to satisfy writer base interface. Unused by this syntax.
        //! \return
        QString writeAfterPart() override;

        //! \brief writeAfterLayer - Implemented to satisfy writer base interface. Unused by this syntax.
        //! \return
        QString writeAfterLayer() override;

        //! \brief writeShutdown - Implemented to satisfy writer base interface. Unused by this syntax.
        //! \return
        QString writeShutdown() override;

        //! \brief writeSettingsFooter
        //! \return the string "EOF"
        QString writeSettingsFooter() override;
        QString writeDwell(Time time) override;

        //! \brief writeIsland
        //! \param island: all polygons in an island
        //! \param islandZvalue: z value of island
        //! \return DXF fragment representing island
        QString writeIsland(PolygonList island, float island_z_value);

        //! \brief writeLayerOffsets
        //! \param origin
        //! \param destination
        //! \return DXF 999 comment with the origin and destination positions of each island for a robotic arm to stack
        QString writeLayerOffsets(QVector<Point> origins, QVector<Point> destinations, float origin_z_value, QVector<float> destination_z_values);
    };
}

#endif // SHEET_LAMINATION_WRITER_H
