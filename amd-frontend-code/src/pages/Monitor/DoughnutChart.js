import React, { useEffect, useState, useRef } from "react";
import { Doughnut } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import Typography from "@mui/material/Typography";

// Register the necessary Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

// Function to generate dynamic colors
const generateColors = (count) => {
  const colors = ["#1034A6", "#01DCE7", "#367588", "#1E90FF", "#001F3F"];
  return Array.from({ length: count }, (_, i) => colors[i % colors.length]);
};

const DoughnutChart = ({ percentages, labels }) => {
  const [activeSegment, setActiveSegment] = useState({
    index: 0,
    label: labels[0],
    value: percentages[0],
  });
  const [hoveredSegment, setHoveredSegment] = useState(null);

  const chartRef = useRef(null);

  // Generate colors based on the number of labels
  const backgroundColors = generateColors(labels.length);
  const hoverBackgroundColors = generateColors(labels.length);

  const data = {
    labels: labels,
    datasets: [
      {
        data: percentages,
        backgroundColor: backgroundColors,
        hoverBackgroundColor: hoverBackgroundColors,
      },
    ],
  };

  const options = {
    cutout: "75%",
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: false,
      },
      title: {
        display: true,
        text: "Generation by Categories",
        font: {
          size: 12,
          weight: "bold",
        },
        padding: {
          top: 10,
          bottom: 25,
        },
      },
    },
    onHover: (event, elements, chart) => {
      if (elements && elements.length) {
        const segment = elements[0];
        setHoveredSegment(segment.index);
        setActiveSegment({
          index: segment.index,
          label: data.labels[segment.index],
          value: data.datasets[0].data[segment.index],
        });
      } else {
        setHoveredSegment(null);
      }
      chart.update(); // Force chart update on hover
    },
  };

  const centerText = (
    <>
      <Typography variant="h5" component="div">
        {activeSegment.label}
      </Typography>
      <Typography variant="body1">{activeSegment.value}</Typography>
    </>
  );

  useEffect(() => {
    const chart = chartRef.current;
    if (chart) {
      chart.update();
    }
  }, [hoveredSegment, activeSegment]);

  return (
    <div
      style={{
        height: "280px",
        width: "300px",
        margin: "10px",
        position: "relative",
        borderRadius: "10px",
        // boxShadow: "0 2px 4px rgba(0, 0, 0, 0.2)",
        padding: "16px",
      }}
    >
      <Doughnut
        ref={chartRef}
        data={data}
        options={options}
        style={{ height: "100%", width: "100%" }}
      />
      <div
      className="custom-DotnetText"
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          textAlign: "center",
        }}
      >
        <p>

        </p> 

        {centerText}             
      </div>
    </div>
  );
};

export default DoughnutChart;
