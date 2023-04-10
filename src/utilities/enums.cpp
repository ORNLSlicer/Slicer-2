#include "utilities/enums.h"

namespace ORNL
{
    void to_json(json& j, const InfillPatterns& i)
    {
        j = json{{"infill_pattern", static_cast< int >(i)}};
    }

    void from_json(const json& j, InfillPatterns& i)
    {
        i = static_cast< InfillPatterns >(j["infill_pattern"].get< int >());
    }

    void to_json(json& j, const SkeletonInput& i)
    {
        j = json{{"skeleton_input", static_cast<int>(i)}};
    }

    void from_json(const json& j, SkeletonInput& i)
    {
        i = static_cast<SkeletonInput>(j["skeleton_input"].get<int>());
    }

    void to_json(json& j, const IslandOrderOptimization& o)
    {
        j = json{{"island_order_optimization", static_cast< int >(o)}};
    }

    void from_json(const json& j, IslandOrderOptimization& o)
    {
        o = static_cast< IslandOrderOptimization >(j["island_order_optimization"].get< int >());
    }

    void to_json(json& j, const PathOrderOptimization& o)
    {
        j = json{{"path_order_optimization", static_cast< int >(o)}};
    }

    void from_json(const json& j, PathOrderOptimization& o)
    {
        o = static_cast< PathOrderOptimization >(j["path_order_optimization"].get< int >());
    }

    void to_json(json& j, const SlicerType& i)
    {
        j = json{{"slicer_type", static_cast<int>(i)}};
    }

    void from_json(const json& j, SlicerType& i)
    {
        i = static_cast<SlicerType>(j["slicer_type"].get<int>());
    }


}  // namespace ORNL
