#ifndef PATHMODIFIER_H
#define PATHMODIFIER_H

// Local
#include "geometry/path.h"
#include "geometry/segments/travel.h"

namespace ORNL {

    class PathModifierGenerator
    {
        public:
           // static void GenerateWait(Path& path, Time waitTime);
            static void GeneratePreStart(Path &path, Distance prestartDistance, Velocity prestartSpeed, AngularVelocity prestartExtruderSpeed, QVector<Path>& outerPath);

            static void GenerateInitialStartup(Path& path, Distance startDistance, Velocity startSpeed, AngularVelocity extruderSpeed, bool enableWidthHeight, double areaMultiplier);
            static void GenerateInitialStartupWithRampUp(Path& path, Distance startDistance, Velocity startSpeed, Velocity endSpeed, AngularVelocity startExtruderSpeed, AngularVelocity endExtruderSpeed, int steps, bool enableWidthHeight, double areaMultiplier);

            static void GenerateSlowdown(Path& path, Distance slowDownDistance, Distance slowDownLiftDistance, Distance slowDownCutoffDistance, Velocity slowDownSpeed, AngularVelocity extruderSpeed, bool enableWidthHeight, double areaMultiplier);

            //! \brief introduce lead-in point at the beginning of each layer
            static void GenerateLayerLeadIn(Path& path, Point leadIn, QSharedPointer<SettingsBase> sb);

            //! \brief based on angle between two segments of motion / path, use configured speed and distance and introduce slow down
            static void GenerateTrajectorySlowdown(Path& path, QSharedPointer<SettingsBase> sb);

            //tip wipe for perimeter/inset forward, backward, and angled
            static void GenerateTipWipe(Path &path, PathModifiers modifiers, Distance wipeDistance, Velocity wipeSpeed, Angle wipeAngle);
            //tip wipe for skin/infill open patterns that connect to inset
            static void GenerateTipWipe(Path &path, PathModifiers modifiers, Distance wipeDistance, Velocity wipeSpeed, QVector<Path>& outerPath, Angle wipeAngle);
            //forward tip wipe for open loop paths (infill and skeletons)
            static void GenerateForwardTipWipeOpenLoop(Path &path, PathModifiers modifiers, Distance wipeDistance, Velocity wipeSpeed);

            static void GenerateSpiralLift(Path& path, Distance spiralWidth, Distance spiralHeight, int spiralPoints, Velocity spiralLiftVelocity, bool supportsG3);

        private:
            static void writeSegment(Path& path, Point start, Point end, Distance width, Distance height, Velocity speed, Acceleration acceleration,
                                     AngularVelocity extruder_speed, RegionType regionType, PathModifiers pathModifiers, int materialNumber, QVector<int> extruders);
            static void writeArcSegment(Path& path, Point start, Point end, Point center, Angle angle, bool ccw, Distance width, Distance height, Velocity speed, Acceleration acceleration,
                                     AngularVelocity extruder_speed, RegionType regionType, PathModifiers pathModifiers, int materialNumber, QVector<int> extruders);

            //! \brief use configured speed and distance and mark ramping
            static void GenerateRamp(Path& path, bool& segmentSplitted, int segmentIndex, PathModifiers pathModifiers,
                                     Distance rampLength, Velocity speed, AngularVelocity extruderSpeed);

    };
}  // namespace ORNL

#endif  // PATHMODIFIER_H
