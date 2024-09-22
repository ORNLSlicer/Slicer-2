// Header
#include <QFile>
#include <QTextStream>
#include <QStringBuilder>
#include <QRegularExpression>
#include <QDir>
#include <QStringList>

#include <geometry/point.h>

#include "managers/settings/settings_manager.h"
#include "threading/gcode_simulation_output.h"

namespace ORNL
{
GCodeSimulationOutput::GCodeSimulationOutput(QString tempLocation, QString path, QString filename, QString text, GcodeMeta meta) :
    m_temp_location(tempLocation), m_path(path), m_filename(filename), m_text(text), m_selected_meta(meta)
{
    //NOP
}

void GCodeSimulationOutput::run()
{
    QChar comma(','), newline('\n'), space(' '), x('X'), y('Y'), z('Z'), w('W'), f('F'), s('S'), zero('0');
    qint16 layerNum = 0;
    QStringList lines = m_text.split(newline);
    QString G0("G0"), G1("G1"), M3("M3"), M5("M5"), M64("M64"), commaSpace(", ");
    QString xval("0"), yval("0"), zval, wval, sval("0"), extruding("0"), velocity;
    QString rapidVelocity = QString::number(GSM->getGlobal()->setting<Velocity>(Constants::ProfileSettings::Travel::kSpeed).to(m_selected_meta.m_velocity_unit), 'f', 4);
    zval = QString::number(GSM->getGlobal()->setting<Distance>(Constants::PrinterSettings::Dimensions::kZMax).to(m_selected_meta.m_distance_unit), 'f', 4);
    wval = QString::number(GSM->getGlobal()->setting<Distance>(Constants::PrinterSettings::Dimensions::kWMax).to(m_selected_meta.m_distance_unit), 'f', 4);

    currentX = 0;
    currentY = 0;
    currentZ = zval.toDouble();
    currentW = wval.toDouble();
    currentTime = 0;
    useMetric = true;
    isG0 = false;

    QFile tempFile(m_temp_location % "temp");
    if (tempFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
    {
        QTextStream out(&tempFile);
        out << m_text;
        tempFile.close();

        QFile::rename(tempFile.fileName(), m_filename);
    }

    QFileInfo fi(m_filename);
    QString filePath = fi.absolutePath() + "\\" + fi.baseName() + "_simulation_output.txt";

    QFile file(filePath);
    file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text);
    QTextStream out(&file);

    QString line;

    for(int i=0; i<lines.size(); i++)
    {
        line = lines[i];
        sval = "1"; // reset sval

        // Cincinnati uses inches and inches/minute
        if(line.contains("Cincinnati"))
        {
            useMetric = false;
        }

        // Extrusion on/off needs to appear in the simulation txt file 1 line before it happens in the g-code\
        // Look at the next line to determine the proper value for the "extruding" flag
        if(i + 1 < lines.size() && lines[i+1].startsWith(G1)) //Check next motion to see if extrusion turns off with S0
        {
            QString temp = lines[i+1].mid(0, line.indexOf(m_selected_meta.m_comment_starting_delimiter));
            QVector<QStringRef> params = temp.splitRef(space);

            if(params[0] == G1)
            {
                for(int i = 1, end = params.size(); i < end; ++i)
                {
                    if(params[i].startsWith(s))
                        sval = params[i].mid(1).toString();
                }
                isG0 = false;
            }

            if(sval == "0" || sval == "0.0000")
            {
                extruding = "0";
            }
        }
        else if(i + 1 < lines.size() && (lines[i+1].startsWith(M3) || lines[i+1].startsWith(M64)))
        {
            extruding = "1";
        }
        else if(i + 1 < lines.size() && lines[i+1].startsWith(M5))
        {
            extruding = "0";
        }

        // Look at current line to creat the output
        if(line.startsWith(G0))
        {
            QString temp = line.mid(0, line.indexOf(m_selected_meta.m_comment_starting_delimiter));
            QVector<QStringRef> params = temp.splitRef(space);

            if(params[0] == G0)
            {
                for(int i = 1, end = params.size(); i < end; ++i)
                {
                    if(params[i].startsWith(x))
                        xval = params[i].mid(1).toString();
                    else if(params[i].startsWith(y))
                        yval = params[i].mid(1).toString();
                    else if(params[i].startsWith(z))
                        zval = params[i].mid(1).toString();
                    else if(params[i].startsWith(w))
                        wval = params[i].mid(1).toString();
                }
                isG0 = true;
            }

            calculateTime(xval, yval, zval, wval, rapidVelocity);
            QString tempt = MathUtils::formattedTimeSpan(currentTime());

            out << currentTime() << ", " % xval % ", " % yval % ", " << currentZ() << ", " % extruding % newline;
        }
        else if(line.startsWith(G1))
        {
            QString temp = line.mid(0, line.indexOf(m_selected_meta.m_comment_starting_delimiter));
            QVector<QStringRef> params = temp.splitRef(space);

            if(params[0] == G1)
            {
                for(int i = 1, end = params.size(); i < end; ++i)
                {
                    if(params[i].startsWith(x))
                        xval = params[i].mid(1).toString();
                    else if(params[i].startsWith(y))
                        yval = params[i].mid(1).toString();
                    else if(params[i].startsWith(z))
                        zval = params[i].mid(1).toString();
                    else if(params[i].startsWith(w))
                        wval = params[i].mid(1).toString();
                    else if(params[i].startsWith(f))
                        velocity = params[i].mid(1).toString();
                    else if(params[i].startsWith(s))
                        sval = params[i].mid(1).toString();
                }
                isG0 = false;
            }

            if(sval == "0" || sval == "0.0000")
            {
                extruding = "0";
            }

            calculateTime(xval, yval, zval, wval, velocity);

            out << currentTime() << ", " % xval % ", " % yval % ", " << currentZ() << ", " % extruding % newline;
        }
    }

    file.close();
}

void GCodeSimulationOutput::calculateTime(QString X, QString Y, QString Z, QString W, QString F)
{
    Distance dist = 0;
    Distance tempX = X.toDouble();
    Distance tempY = Y.toDouble();
    Distance tempZ = Z.toDouble();
    Distance tempW = W.toDouble();
    Velocity tempF = F.toDouble();
    Velocity tempFseconds;
    Distance dx = tempX - currentX;
    Distance dy = tempY - currentY;
    Distance dz = tempZ - currentZ;
    Distance dw = tempW - currentW;
    Time tempTime = 0;

    // Set acceleration default value
    Acceleration currentAccel;
    // Metric
    currentAccel = Acceleration(1500); // units mm/s^2 - this is ~0.153g for JuggerBot3D
    // Imperial
    if (!useMetric)
    {
        currentAccel = Acceleration(23.165); // units in/s^2 - this is ~0.06g for CI
    }

    // Calculate distance to be used for time calculation
    // If only W moves, use W speed and acceleration. Typically happens with a G0
    if (dx == 0 && dy == 0 && dz == 0 && dw != 0)
    {
        dist = dw;
        if (isG0)
        {
            tempF = GSM->getGlobal()->setting<Velocity>(Constants::PrinterSettings::MachineSpeed::kWTableSpeed).to(m_selected_meta.m_velocity_unit);
        }
        // Update acceleration for W table moves which only exist on the CI syntax, so units are in/s^2
        currentAccel = Acceleration(7.87);
    }
    // If only Z moves, use Z speed and acceleration. Typically happens with a G0
    else if (dx == 0 && dy == 0 && dz != 0 && dw == 0)
    {
        dist = dz;
        if (isG0)
        {
            tempF = GSM->getGlobal()->setting<Velocity>(Constants::PrinterSettings::MachineSpeed::kZSpeed).to(m_selected_meta.m_velocity_unit);
        }
        // For Z only moves, set acceleration to 200mm/s^2 or 7.87in/s*2
        if (!useMetric)
        {
            currentAccel = Acceleration(7.87); // units in/s^2
        }
        else
        {
            currentAccel = Acceleration(200); // units mm/s^2
        }
    }
    // Predominantly X and Y motion
    else
    {
        dist = sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
    }

    dist = abs(dist);
    tempFseconds = tempF / 60.0;

    Time accelTime = tempFseconds / currentAccel;
    Distance accelDist = (tempFseconds * tempFseconds) / (2.0 * currentAccel);

    if(dist > 2.0 * accelDist)
    {
        tempTime = 2 * accelTime + (dist - 2.0 * accelDist) / tempFseconds;
    }
    else
    {
        tempTime = sqrt(2.0 * dist / currentAccel);
    }

    // Add time to the total time
    currentTime += tempTime;

    // Update Current Values
    currentX = tempX;
    currentY = tempY;
    currentZ = tempZ + abs(tempW);
    currentW = tempW;
}

}  // namespace ORNL
