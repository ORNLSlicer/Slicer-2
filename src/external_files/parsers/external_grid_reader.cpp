// Header
#include <QFile>

#include "external_files/parsers/external_grid_reader.h"
#include "units/unit.h"

#include "xlsxdocument.h"

namespace ORNL
{
    ExternalGridReader::ExternalGridReader(QString file) : m_filename(file)
    {
        //NOP
    }

    void ExternalGridReader::run()
    {
        try
        {
            QXlsx::Document xlsx(m_filename);

            double xMin, yMin, zMin, xMax = INT_MIN, yMax = INT_MIN, zMax = INT_MIN;
            double xStep = -1, yStep = -1, zStep = -1;
            int totalRows = 3;
            bool rowFilled = true, stepsIdentified = false;
            Distance d;
            while(rowFilled)
            {
                if(!xlsx.cellAt(totalRows, 1))
                    rowFilled = false;
                else
                {
                    if(!stepsIdentified)
                    {
                        bool isDouble;
                        double x = xlsx.cellAt(totalRows, 2)->value().toDouble(&isDouble);
                        if(!isDouble)
                            throw InvalidParseException("Grid locations are not valid numbers");

                        Distance xD = d.from(x, inch);
                        x = xD();

                        double y = xlsx.cellAt(totalRows, 3)->value().toDouble(&isDouble);
                        if(!isDouble)
                            throw InvalidParseException("Grid locations are not valid numbers");

                        Distance yD = d.from(y, inch);
                        y = yD();

                        double z = xlsx.cellAt(totalRows, 4)->value().toDouble(&isDouble);
                        if(!isDouble)
                            throw InvalidParseException("Grid locations are not valid numbers");

                        Distance zD = d.from(z, inch);
                        z = zD();

                        if(x > xMax)
                            xMax = x;
                        if(y > yMax)
                            yMax = y;
                        if(z > zMax)
                            zMax = z;

                        if(totalRows == 3)
                        {
                            xMin = x;
                            yMin = y;
                            zMin = z;
                        }
                        else
                        {
                            if(xStep == -1)
                            {
                                if(x != xMin)
                                    xStep = x - xMin;
                            }
                            if(yStep == -1)
                            {
                                if(y != yMin)
                                    yStep = y - yMin;
                            }
                            if(zStep == -1)
                            {
                                if(z != zMin)
                                    zStep = z - zMin;
                            }
                            if(xStep != -1 && yStep != -1 && zStep != -1)
                                stepsIdentified = true;
                        }
                    }
                    ++totalRows;
                }
            }

            if(xStep == -1)
                xStep = xMin;
            if(yStep == -1)
                yStep = yMin;
            if(zStep == -1)
                zStep = zMin;

            int totalLayers = ((zMax - zMin) / zStep) + 1;
            if(totalLayers <= 0)
                throw InvalidParseException("No valid layers");

            QVector<RecipeMap> recipeMaps;
            QString recipeKey = "RECIPE MATRIX";
            bool foundRecipeMap = false;
            for(int i = totalRows, end = xlsx.dimension().lastRow(); i <= end; ++i)
            {
                if(xlsx.cellAt(i, 1))
                {
                    if(xlsx.cellAt(i, 1)->value().toString().toUpper() == recipeKey)
                    {
                        foundRecipeMap = true;
                        ++i;
                    }
                    else if(foundRecipeMap)
                    {
                        if(!xlsx.cellAt(i, 1))
                            break;

                        bool isNum;
                        int id = xlsx.cellAt(i, 1)->value().toInt(&isNum);
                        if(!isNum)
                            throw InvalidParseException("Recipe id must be an integer");

                        double min = xlsx.cellAt(i, 2)->value().toDouble(&isNum);
                        if(!isNum)
                            throw InvalidParseException("Recipe minimum must be a valid double");

                        double max = xlsx.cellAt(i, 3)->value().toInt(&isNum);
                        if(!isNum)
                            throw InvalidParseException("Recipe maximum must be an double");

                        for(RecipeMap rMap : recipeMaps)
                        {
                            if(rMap.m_id == id)
                                throw InvalidParseException("Recipe id declared more than once");

                            if(min >= rMap.m_min && min <= rMap.m_max)
                                throw InvalidParseException("Recipe min cannot overlap with other ranges");

                            if(max <= rMap.m_max && max >= rMap.m_min)
                                throw InvalidParseException("Recipe max cannot overlap with other ranges");
                        }

                        recipeMaps.append(RecipeMap(id, min, max));
                    }
                }
            }

            ExternalGridInfo gridInfo(zMin, zStep);
            gridInfo.m_grid_layers.resize(totalLayers);

            for (int row = 3; row < totalRows; ++row)
            {
                Distance xD = d.from(xlsx.cellAt(row, 2)->value().toFloat(), inch);
                Distance yD = d.from(xlsx.cellAt(row, 3)->value().toFloat(), inch);
                Distance zD = d.from(xlsx.cellAt(row, 4)->value().toFloat(), inch);

                double x = xD();
                double y = yD();
                double z = zD();

                QString densityStr = xlsx.cellAt(row, 6)->value().toString();
                densityStr.chop(1);
                double density = densityStr.toDouble();

                int key = (z - zMin) / zStep;
                int columnIndex = (x - xMin) / xStep;
                int rowIndex = (y - yMin) / yStep;

                if(!gridInfo.m_grid_layers[key].isSet())
                    gridInfo.m_grid_layers[key].setParams(xMin, xStep, xMax, yMin, yStep, yMax, recipeMaps);

                gridInfo.m_grid_layers[key].m_grid[columnIndex][rowIndex] = density;

                emit statusUpdate((double)(row + 1) / (double)totalRows * 100);
            }
            emit gridFileProcessed(gridInfo);
        }
        catch(ExceptionBase eb)
        {
            emit gridFailed(QString(eb.what()));
        }
    }

}  // namespace ORNL
