// Main Module
#include "step/layer/regions/skin.h"
#include <geometry/segments/line.h>
#include "geometry/path_modifier.h"
#include "geometry/pattern_generator.h"
#include "geometry/curve_fitting.h"

namespace ORNL {
    Skin::Skin(const QSharedPointer<SettingsBase>& sb, const int index, const QVector<SettingsPolygon>& settings_polygons, const SingleExternalGridInfo& gridInfo)
        : RegionBase(sb, index, settings_polygons, gridInfo)
    {
        // NOP
    }

    QString Skin::writeGCode(QSharedPointer<WriterBase> writer) {
        QString gcode;
        gcode += writer->writeBeforeRegion(RegionType::kSkin);
        for (Path path : m_paths) {
            gcode += writer->writeBeforePath(RegionType::kSkin);
            for (QSharedPointer<SegmentBase> segment : path.getSegments()) {
                gcode += segment->writeGCode(writer);
            }
            gcode += writer->writeAfterPath(RegionType::kSkin);
        }
        gcode += writer->writeAfterRegion(RegionType::kSkin);
        return gcode;
    }

    void Skin::compute(uint layer_num, QSharedPointer<SyncManager>& sync) {
        m_paths.clear();

        setMaterialNumber(m_sb->setting<int>(Constants::MaterialSettings::MultiMaterial::kSkinNum));

        Distance overlap = m_sb->setting<Distance>(Constants::ProfileSettings::Skin::kOverlap);
        m_geometry = m_geometry.offset(overlap);

        Distance beadWidth = m_sb->setting<Distance>(Constants::ProfileSettings::Skin::kBeadWidth);
        Angle patternAngle = m_sb->setting<Angle>(Constants::ProfileSettings::Skin::kAngle);

        int top_count = m_sb->setting<int>(Constants::ProfileSettings::Skin::kTopCount);
        int bottom_count = m_sb->setting<int>(Constants::ProfileSettings::Skin::kBottomCount);
        int gradual_count = m_sb->setting<int>(Constants::ProfileSettings::Skin::kInfillSteps);

        //! If skin region belongs to top or bottom layer there is no need to compute top or bottom skin
        if (!(top_count > 0 && m_upper_geometry.isEmpty()) || !(bottom_count > 0 && m_lower_geometry.isEmpty()))
        {
            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    computeTopSkin(top_count);
                }

                #pragma omp section
                {
                    computeBottomSkin(bottom_count);
                }
            }
        }
        if(m_sb->setting<bool>(Constants::ProfileSettings::Skin::kInfillEnable))
            computeGradualSkinSteps(gradual_count);

        bool anyGeometry = false;
        if (!m_skin_geometry.isEmpty())
        {
            m_geometry -= m_skin_geometry;
            PolygonList skin_offset = m_skin_geometry.offset(-beadWidth / 2);
            m_computed_geometry = createPatternForArea(static_cast<InfillPatterns>(m_sb->setting<int>(Constants::ProfileSettings::Skin::kPattern)),
                                                       skin_offset, beadWidth, beadWidth, patternAngle);
            anyGeometry = true;
        }

        if(!m_gradual_skin_geometry.isEmpty())
        {
            Angle infillPatternAngle = m_sb->setting<Angle>(Constants::ProfileSettings::Skin::kInfillAngle);
            InfillPatterns gradualPattern = static_cast<InfillPatterns>(m_sb->setting<int>(Constants::ProfileSettings::Skin::kInfillPattern));
            double percentage = 0;
            if(m_sb->setting<bool>(Constants::ProfileSettings::Infill::kEnable))
                percentage = m_sb->setting<double>(Constants::ProfileSettings::Infill::kBeadWidth) / m_sb->setting<double>(Constants::ProfileSettings::Infill::kLineSpacing);

            double densityStep = (1.0 - percentage) / (gradual_count + 1);
            for(int i = 0, end = m_gradual_skin_geometry.size(); i < end; ++i)
            {
                if(!m_gradual_skin_geometry[i].isEmpty())
                {
                    m_geometry -= m_gradual_skin_geometry[i];
                    PolygonList skin_offset = m_gradual_skin_geometry[i].offset(-beadWidth / 2);
                    m_gradual_computed_geometry.push_back(createPatternForArea(gradualPattern, skin_offset,
                                                                               beadWidth, beadWidth / (percentage + densityStep * (end - i)), infillPatternAngle));
                }
            }
            anyGeometry = true;
        }
    }

    QVector<Polyline> Skin::createPatternForArea(InfillPatterns pattern, PolygonList& geometry, Distance beadWidth,
                                                 Distance lineSpacing, Angle patternAngle)
    {
        switch(pattern)
        {
        case InfillPatterns::kLines:
            return PatternGenerator::GenerateLines(geometry, lineSpacing, patternAngle);
        case InfillPatterns::kGrid:
            return PatternGenerator::GenerateGrid(geometry, lineSpacing, patternAngle);
        case InfillPatterns::kConcentric:
        case InfillPatterns::kInsideOutConcentric:
            return PatternGenerator::GenerateConcentric(geometry, beadWidth, lineSpacing);
        case InfillPatterns::kTriangles:
            return PatternGenerator::GenerateTriangles(geometry, lineSpacing, patternAngle);
        case InfillPatterns::kHexagonsAndTriangles:
            return PatternGenerator::GenerateHexagonsAndTriangles(geometry, lineSpacing, patternAngle);
        case InfillPatterns::kHoneycomb:
            return PatternGenerator::GenerateHoneyComb(geometry, beadWidth, lineSpacing, patternAngle);
        }
    }

    void Skin::computeTopSkin(const int& top_count)
    {
        PolygonList temp_geometry = m_geometry;

        //! If skin is within top_count of top layer, compute common geometry
        if (m_upper_geometry_includes_top && m_upper_geometry.size() < top_count)
            for (PolygonList poly : m_upper_geometry)
                temp_geometry &= poly;
        else
            temp_geometry.clear();

        //! Compute difference geometry
        for (PolygonList poly : m_upper_geometry)
            temp_geometry += m_geometry - poly;

        m_skin_geometry += temp_geometry;
    }

    void Skin::computeBottomSkin(const int& bottom_count)
    {
        PolygonList temp_geometry = m_geometry;

        //! If skin is within bottom_count of botton layer, compute common geometry
        if (m_lower_geometry_includes_bottom && m_lower_geometry.size() < bottom_count)
            for (PolygonList poly : m_lower_geometry)
                temp_geometry &= poly;
        else
            temp_geometry.clear();

        //! Compute difference geometry
        for (PolygonList poly : m_lower_geometry)
            temp_geometry += m_geometry - poly;

        m_skin_geometry += temp_geometry;
    }

    void Skin::computeGradualSkinSteps(const int &gradual_count)
    {
        PolygonList currentGradual;
        for (PolygonList poly : m_gradual_geometry)
        {
            m_gradual_skin_geometry.push_back(m_geometry - m_skin_geometry - currentGradual - poly);
            currentGradual += m_gradual_skin_geometry[m_gradual_skin_geometry.size() - 1];
        }
        if(m_gradual_geometry_includes_top && m_gradual_geometry.size() < gradual_count)
        {
            m_gradual_skin_geometry.push_back(m_geometry - m_skin_geometry - currentGradual);

            while(m_gradual_skin_geometry.size() < gradual_count)
                m_gradual_skin_geometry.push_back(PolygonList());
        }
    }

    void Skin::optimize(int layerNumber, Point& current_location, QVector<Path>& innerMostClosedContour, QVector<Path>& outerMostClosedContour, bool& shouldNextPathBeCCW)
    {
        PolylineOrderOptimizer poo(current_location, layerNumber);

        PathOrderOptimization pathOrderOptimization = static_cast<PathOrderOptimization>(
                    this->getSb()->setting<int>(Constants::ProfileSettings::Optimizations::kPathOrder));
        if(pathOrderOptimization == PathOrderOptimization::kCustomPoint)
        {
            Point startOverride(getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPathXLocation),
                                getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPathYLocation));

            poo.setStartOverride(startOverride);
        }

        PointOrderOptimization pointOrderOptimization = static_cast<PointOrderOptimization>(
                    this->getSb()->setting<int>(Constants::ProfileSettings::Optimizations::kPointOrder));

        if(pointOrderOptimization == PointOrderOptimization::kCustomPoint)
        {
            Point startOverride(getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPointXLocation),
                                getSb()->setting<double>(Constants::ProfileSettings::Optimizations::kCustomPointYLocation));

            poo.setStartPointOverride(startOverride);
        }

        poo.setPointParameters(pointOrderOptimization, getSb()->setting<bool>(Constants::ProfileSettings::Optimizations::kMinDistanceEnabled),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kMinDistanceThreshold),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kConsecutiveDistanceThreshold),
                               getSb()->setting<bool>(Constants::ProfileSettings::Optimizations::kLocalRandomnessEnable),
                               getSb()->setting<Distance>(Constants::ProfileSettings::Optimizations::kLocalRandomnessRadius));

        m_paths.clear();
        bool supportsG3 = m_sb->setting<bool>(Constants::PrinterSettings::MachineSetup::kSupportG3);
        InfillPatterns skinPattern = static_cast<InfillPatterns>(m_sb->setting<int>(Constants::ProfileSettings::Skin::kPattern));
        optimizeHelper(poo, supportsG3, innerMostClosedContour, current_location,
                       skinPattern, m_computed_geometry, m_skin_geometry);

        InfillPatterns gradualPattern = InfillPatterns::kLines;
        for(int i = 0, end = m_gradual_computed_geometry.size(); i < end; ++i)
        {
            optimizeHelper(poo, supportsG3, innerMostClosedContour, current_location,
                           gradualPattern, m_gradual_computed_geometry[i], m_gradual_skin_geometry[i]);
        }
    }

    void Skin::optimizeHelper(PolylineOrderOptimizer poo, bool supportsG3, QVector<Path> &innerMostClosedContour,
                              Point &current_location, InfillPatterns pattern, QVector<Polyline> lines, PolygonList geometry)
    {
        poo.setInfillParameters(pattern, geometry, getSb()->setting<Distance>(Constants::ProfileSettings::Skin::kMinPathLength),
                                getSb()->setting<Distance>(Constants::ProfileSettings::Travel::kMinLength));

        poo.setGeometryToEvaluate(lines, RegionType::kSkin, static_cast<PathOrderOptimization>(m_sb->setting<int>(Constants::ProfileSettings::Optimizations::kPathOrder)));

        QVector<Polyline> previouslyLinkedLines;
        while(poo.getCurrentPolylineCount() > 0)
        {
            Polyline result = poo.linkNextPolyline(previouslyLinkedLines);
            if(result.size() > 0)
            {
                Path newPath = createPath(result);
                if(newPath.size() > 0)
                {
                    // Only fit arcs if the infill was concentric
                    if(static_cast<InfillPatterns>(m_sb->setting<int>(Constants::ProfileSettings::Skin::kPattern)) == InfillPatterns::kConcentric)
                    {
                        if(m_sb->setting<bool>(Constants::ExperimentalSettings::CurveFitting::kEnableArcFitting) ||
                                m_sb->setting<bool>(Constants::ExperimentalSettings::CurveFitting::kEnableSplineFitting))
                            CurveFitting::Fit(newPath, m_sb);
                    }


                    calculateModifiers(newPath, m_sb->setting<bool>(Constants::PrinterSettings::MachineSetup::kSupportG3), innerMostClosedContour);
                    PathModifierGenerator::GenerateTravel(newPath, current_location, m_sb->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed));
                    current_location = newPath.back()->end();
                    previouslyLinkedLines.push_back(result);
                    m_paths.push_back(newPath);
                }
            }
        }
    }

    Path Skin::createPath(Polyline line) {

        Distance width                  = m_sb->setting< Distance >(Constants::ProfileSettings::Skin::kBeadWidth);
        Distance height                 = m_sb->setting< Distance >(Constants::ProfileSettings::Layer::kLayerHeight);
        Velocity speed                  = m_sb->setting< Velocity >(Constants::ProfileSettings::Skin::kSpeed);
        Acceleration acceleration       = m_sb->setting< Acceleration >(Constants::PrinterSettings::Acceleration::kInfill);
        AngularVelocity extruder_speed  = m_sb->setting< AngularVelocity >(Constants::ProfileSettings::Skin::kExtruderSpeed);
        int material_number             = m_sb->setting< int >(Constants::MaterialSettings::MultiMaterial::kSkinNum);

        Path newPath;
        for (int i = 0; i < line.size() - 1; i++)
        {
            QSharedPointer<LineSegment> line_segment = QSharedPointer<LineSegment>::create(line[i], line[i + 1]);

            line_segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            width);
            line_segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           height);
            line_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            speed);
            line_segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            acceleration);
            line_segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    extruder_speed);
            line_segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   material_number);
            line_segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kSkin);

            newPath.append(line_segment);
        }

        //! Creates closing segment if infill pattern is concentric
        if (static_cast<InfillPatterns>(m_sb->setting<int>(Constants::ProfileSettings::Skin::kPattern)) == InfillPatterns::kConcentric ||
                static_cast<InfillPatterns>(m_sb->setting<int>(Constants::ProfileSettings::Skin::kPattern)) == InfillPatterns::kInsideOutConcentric)
        {
            QSharedPointer<LineSegment> line_segment = QSharedPointer<LineSegment>::create(line.last(), line.first());

            line_segment->getSb()->setSetting(Constants::SegmentSettings::kWidth,            width);
            line_segment->getSb()->setSetting(Constants::SegmentSettings::kHeight,           height);
            line_segment->getSb()->setSetting(Constants::SegmentSettings::kSpeed,            speed);
            line_segment->getSb()->setSetting(Constants::SegmentSettings::kAccel,            acceleration);
            line_segment->getSb()->setSetting(Constants::SegmentSettings::kExtruderSpeed,    extruder_speed);
            line_segment->getSb()->setSetting(Constants::SegmentSettings::kMaterialNumber,   material_number);
            line_segment->getSb()->setSetting(Constants::SegmentSettings::kRegionType,       RegionType::kSkin);

            newPath.append(line_segment);
        }

        return newPath;
    }

    void Skin::addUpperGeometry(const PolygonList& poly_list) {
        m_upper_geometry.push_back(poly_list);
    }

    void Skin::addLowerGeometry(const PolygonList& poly_list) {
        m_lower_geometry.push_back(poly_list);
    }

    void Skin::addGradualGeometry(const PolygonList &poly_list) {
        m_gradual_geometry.push_back(poly_list);
    }

    void Skin::setGeometryIncludes(bool top, bool bottom, bool gradual) {
        m_upper_geometry_includes_top = top;
        m_lower_geometry_includes_bottom = bottom;
        m_gradual_geometry_includes_top = gradual;
    }

    void Skin::calculateModifiers(Path& path, bool supportsG3, QVector<Path>& innerMostClosedContour)
    {
        if(m_sb->setting<bool>(Constants::ExperimentalSettings::Ramping::kTrajectoryAngleEnabled))
        {
            PathModifierGenerator::GenerateTrajectorySlowdown(path, m_sb);
        }

        //add modifiers
        if(m_sb->setting<bool>(Constants::MaterialSettings::Slowdown::kSkinEnable))
        {
            PathModifierGenerator::GenerateSlowdown(path, m_sb->setting<Distance>(Constants::MaterialSettings::Slowdown::kSkinDistance),
                                                    m_sb->setting<Distance>(Constants::MaterialSettings::Slowdown::kSkinLiftDistance),
                                                    m_sb->setting<Distance>(Constants::MaterialSettings::Slowdown::kSkinCutoffDistance),
                                                    m_sb->setting<Velocity>(Constants::MaterialSettings::Slowdown::kSkinSpeed),
                                                    m_sb->setting<AngularVelocity>(Constants::MaterialSettings::Slowdown::kSkinExtruderSpeed),
                                                    m_sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableWidthHeight),
                                                    m_sb->setting<double>(Constants::MaterialSettings::Slowdown::kSlowDownAreaModifier));
        }
        if(m_sb->setting<bool>(Constants::MaterialSettings::TipWipe::kSkinEnable))
        {
            // if Forward OR (if Optimal AND (Perimeter OR Inset)) OR (if Optimal AND (Concentric or Inside Out Concentric))
            if(static_cast<TipWipeDirection>(m_sb->setting<int>(Constants::MaterialSettings::TipWipe::kSkinDirection)) == TipWipeDirection::kForward ||
                    (static_cast<TipWipeDirection>(m_sb->setting<int>(Constants::MaterialSettings::TipWipe::kSkinDirection)) == TipWipeDirection::kOptimal &&
                     (m_sb->setting<int>(Constants::ProfileSettings::Perimeter::kEnable) || m_sb->setting<int>(Constants::ProfileSettings::Inset::kEnable))) ||
                    (static_cast<TipWipeDirection>(m_sb->setting<int>(Constants::MaterialSettings::TipWipe::kSkinDirection)) == TipWipeDirection::kOptimal &&
                                         (static_cast<InfillPatterns>(m_sb->setting<int>(Constants::ProfileSettings::Skin::kPattern)) == InfillPatterns::kConcentric ||
                                          static_cast<InfillPatterns>(m_sb->setting<int>(Constants::ProfileSettings::Skin::kPattern)) == InfillPatterns::kInsideOutConcentric)))
            {
                if(static_cast<InfillPatterns>(m_sb->setting<int>(Constants::ProfileSettings::Skin::kPattern)) == InfillPatterns::kConcentric ||
                        static_cast<InfillPatterns>(m_sb->setting<int>(Constants::ProfileSettings::Skin::kPattern)) == InfillPatterns::kInsideOutConcentric)
                    PathModifierGenerator::GenerateTipWipe(path, PathModifiers::kForwardTipWipe, m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkinDistance),
                                                           m_sb->setting<Velocity>(Constants::MaterialSettings::TipWipe::kSkinSpeed),
                                                           m_sb->setting<Angle>(Constants::MaterialSettings::TipWipe::kSkinAngle),
                                                           m_sb->setting<AngularVelocity>(Constants::MaterialSettings::TipWipe::kSkinExtruderSpeed),
                                                           m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkinLiftHeight),
                                                           m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkinCutoffDistance));
                else if(m_sb->setting<int>(Constants::ProfileSettings::Perimeter::kEnable) || m_sb->setting<int>(Constants::ProfileSettings::Inset::kEnable))
                    PathModifierGenerator::GenerateTipWipe(path, PathModifiers::kForwardTipWipe, m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkinDistance),
                                                           m_sb->setting<Velocity>(Constants::MaterialSettings::TipWipe::kSkinSpeed), innerMostClosedContour,
                                                           m_sb->setting<Angle>(Constants::MaterialSettings::TipWipe::kSkinAngle),
                                                           m_sb->setting<AngularVelocity>(Constants::MaterialSettings::TipWipe::kSkinExtruderSpeed),
                                                           m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkinLiftHeight),
                                                           m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkinCutoffDistance));
                else
                    PathModifierGenerator::GenerateForwardTipWipeOpenLoop(path, PathModifiers::kForwardTipWipe, m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkinDistance),
                                                                          m_sb->setting<Velocity>(Constants::MaterialSettings::TipWipe::kSkinSpeed),
                                                                          m_sb->setting<AngularVelocity>(Constants::MaterialSettings::TipWipe::kSkinExtruderSpeed),
                                                                          m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkinLiftHeight),
                                                                          m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkinCutoffDistance));
            }
            else if(static_cast<TipWipeDirection>(m_sb->setting<int>(Constants::MaterialSettings::TipWipe::kSkinDirection)) == TipWipeDirection::kAngled)
            {
                PathModifierGenerator::GenerateTipWipe(path, PathModifiers::kAngledTipWipe, m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkinDistance),
                                                       m_sb->setting<Velocity>(Constants::MaterialSettings::TipWipe::kSkinSpeed),
                                                       m_sb->setting<Angle>(Constants::MaterialSettings::TipWipe::kSkinAngle),
                                                       m_sb->setting<AngularVelocity>(Constants::MaterialSettings::TipWipe::kSkinExtruderSpeed),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkinLiftHeight),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkinCutoffDistance));
            }
            else
                PathModifierGenerator::GenerateTipWipe(path, PathModifiers::kReverseTipWipe, m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkinDistance),
                                                       m_sb->setting<Velocity>(Constants::MaterialSettings::TipWipe::kSkinSpeed),
                                                       m_sb->setting<Angle>(Constants::MaterialSettings::TipWipe::kSkinAngle),
                                                       m_sb->setting<AngularVelocity>(Constants::MaterialSettings::TipWipe::kSkinExtruderSpeed),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkinLiftHeight),
                                                       m_sb->setting<Distance>(Constants::MaterialSettings::TipWipe::kSkinCutoffDistance));
        }
        if(m_sb->setting<bool>(Constants::MaterialSettings::SpiralLift::kSkinEnable))
        {
            PathModifierGenerator::GenerateSpiralLift(path, m_sb->setting<Distance>(Constants::MaterialSettings::SpiralLift::kLiftRadius),
                                                      m_sb->setting<Distance>(Constants::MaterialSettings::SpiralLift::kLiftHeight),
                                                      m_sb->setting<int>(Constants::MaterialSettings::SpiralLift::kLiftPoints),
                                                      m_sb->setting<Velocity>(Constants::MaterialSettings::SpiralLift::kLiftSpeed), supportsG3);
        }
        if(m_sb->setting<bool>(Constants::MaterialSettings::Startup::kSkinEnable))
        {
            if(m_sb->setting<bool>(Constants::MaterialSettings::Startup::kSkinRampUpEnable))
            {
                PathModifierGenerator::GenerateInitialStartupWithRampUp(path, m_sb->setting<Distance>(Constants::MaterialSettings::Startup::kSkinDistance),
                                                              m_sb->setting<Velocity>(Constants::MaterialSettings::Startup::kSkinSpeed),
                                                              m_sb->setting<Velocity>(Constants::ProfileSettings::Skin::kSpeed),
                                                              m_sb->setting<AngularVelocity>(Constants::MaterialSettings::Startup::kSkinExtruderSpeed),
                                                              m_sb->setting<AngularVelocity>(Constants::ProfileSettings::Skin::kExtruderSpeed),
                                                              m_sb->setting<int>(Constants::MaterialSettings::Startup::kSkinSteps),
                                                              m_sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableWidthHeight),
                                                              m_sb->setting<double>(Constants::MaterialSettings::Startup::kStartUpAreaModifier));
            }
            else
            {
                PathModifierGenerator::GenerateInitialStartup(path, m_sb->setting<Distance>(Constants::MaterialSettings::Startup::kSkinDistance),
                                                              m_sb->setting<Velocity>(Constants::MaterialSettings::Startup::kSkinSpeed),
                                                              m_sb->setting<AngularVelocity>(Constants::MaterialSettings::Startup::kSkinExtruderSpeed),
                                                              m_sb->setting<bool>(Constants::ProfileSettings::SpecialModes::kEnableWidthHeight),
                                                              m_sb->setting<double>(Constants::MaterialSettings::Startup::kStartUpAreaModifier));
            }
        }
        if(m_sb->setting<bool>(Constants::ProfileSettings::Skin::kPrestart))
        {
            if(static_cast<InfillPatterns>(m_sb->setting<int>(Constants::ProfileSettings::Skin::kPattern)) == InfillPatterns::kLines)
            {
                PathModifierGenerator::GeneratePreStart(path, m_sb->setting<Distance>(Constants::ProfileSettings::Skin::kPrestartDistance),
                                                        m_sb->setting<Velocity>(Constants::ProfileSettings::Skin::kPrestartSpeed),
                                                        m_sb->setting<AngularVelocity>(Constants::ProfileSettings::Skin::kPrestartExtruderSpeed), innerMostClosedContour);
            }
        }
    }
}
