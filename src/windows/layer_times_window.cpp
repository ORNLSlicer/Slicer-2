#include "windows/layer_times_window.h"

#include <QIcon>
#include <QLabel>
#include <QStringBuilder>

#include "utilities/mathutils.h"

namespace ORNL
{
  LayerTimesWindow::LayerTimesWindow(QWidget *parent)
  {
     QIcon icon;
     icon.addFile(QStringLiteral(":/icons/slicer2.png"), QSize(), QIcon::Normal, QIcon::Off);
     setWindowIcon(icon);

     m_layout = new QGridLayout();
     QLabel *lblMinLayerTime = new QLabel("Minimum Layer Time (seconds):");
     m_min_layer_time_edit = new QLineEdit();
     m_layout->addWidget(lblMinLayerTime, 0, 0);
     m_layout->addWidget(m_min_layer_time_edit, 0, 1);

     m_layer_times_edit = new QTextEdit();
     m_layer_times_edit->setReadOnly(true);
     m_layout->addWidget(m_layer_times_edit, 1, 0, 2, 2);

     this->setLayout(m_layout);

     setupEvents();
  }

  void LayerTimesWindow::setupEvents()
  {
      connect(m_min_layer_time_edit, &QLineEdit::textChanged, this, &LayerTimesWindow::updateText);
  }

  void LayerTimesWindow::updateTimeInformation(QList<QList<Time>> layer_times, Time min_layer_time, Time max_layer_time, bool adjusted_layer_time)
  {
      m_layer_times = layer_times;
      m_minimum_layer_time = min_layer_time;
      m_maximum_layer_time = max_layer_time;
      m_adjusted_layer_time = adjusted_layer_time;
      m_min = INT_MAX, m_max = INT_MIN;
      m_min_index = -1, m_max_index = -1;
      m_total_time = 0;
      m_total_adjusted_time = 0;

      for(int i = 1; i < m_layer_times.size(); ++i)
      {
          // a layer's time is the max time of any extruders printing on that layer
          Time& current_time = m_layer_times[i][0];
          for (auto& time : m_layer_times[i])
          {
              current_time = qMax(current_time, time);
          }

          if(current_time < m_min)
          {
              m_min_index = i;
              m_min = current_time;
          }

          if(current_time > m_max)
          {
              m_max_index = i;
              m_max = current_time;
          }

          if (current_time < m_minimum_layer_time)
              m_total_adjusted_time += m_minimum_layer_time;
          else if (current_time > m_maximum_layer_time)
              m_total_adjusted_time += m_maximum_layer_time;
          else
              m_total_adjusted_time += current_time;

          m_total_time += current_time;
      }

      updateText();
  }

  void LayerTimesWindow::updateText()
  {
      Time layerTimeThreshold(0);
      bool flag = false;
      layerTimeThreshold = Time(m_min_layer_time_edit->text().toInt(&flag));

      QString layerTimeString = "Total print time: " % MathUtils::formattedTimeSpan(m_total_time()) % "<br>";

      if ((m_minimum_layer_time > 0 || m_maximum_layer_time > 0) && m_adjusted_layer_time)
          layerTimeString = layerTimeString % "Total adjusted time: " % MathUtils::formattedTimeSpan(m_total_adjusted_time()) % "<br>";

      layerTimeString = layerTimeString %
              "Minimum Layer Time Layer #" % QString::number(m_min_index) % ", " % MathUtils::formattedTimeSpanHHMMSS(m_min()) % "<br>"
              "Maximum Layer Time Layer #" % QString::number(m_max_index) % ", " % MathUtils::formattedTimeSpanHHMMSS(m_max()) % "<br>";

      for(int i = 0; i < m_layer_times.count(); ++i)
      {
          // a layer's time is the max time of any extruders printing on that layer
          Time& current_time = m_layer_times[i][0];
          for (auto& time : m_layer_times[i])
          {
              current_time = qMax(current_time, time);
          }

          QString oneLayer = "Layer " % QString::number(i) % " " % MathUtils::formattedTimeSpanHHMMSS(current_time());
          if(i > 0 && current_time > 1 && m_adjusted_layer_time){
              if(m_minimum_layer_time > 0 && current_time < m_minimum_layer_time)
                  oneLayer += " Adjusted " % MathUtils::formattedTimeSpanHHMMSS(m_minimum_layer_time);
              else if(m_maximum_layer_time > 0 && current_time > m_maximum_layer_time)
                  oneLayer += " Adjusted " % MathUtils::formattedTimeSpanHHMMSS(m_maximum_layer_time);
          }

          if(max(current_time, m_minimum_layer_time) < layerTimeThreshold)
          {
              layerTimeString += "<font color=\"red\">" % oneLayer % "</font><br>";
          }
          else
          {
              layerTimeString += oneLayer % "<br>";
          }
      }
      m_layer_times_edit->setText(layerTimeString);
  }
}
