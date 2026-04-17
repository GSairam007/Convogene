import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { Card, CardContent, Typography, Grid, Box } from "@mui/material";
import infobellImg from "../../assets/Images/infobellLogo.png";
import backLogo from "../../assets/Images/cancel.png"; // Import the back logo image
import DoughnutChart from "./DoughnutChart";
import ColumnGroupingTable from "./table_data";
import axios from "axios";
import g1 from "../../assets/Images/g1.svg";
import TotalUser from "../../assets/Images/TotalUser.svg";
import hi from "../../assets/Images/hi.svg";
import ri from "../../assets/Images/ri.svg";
import yi from "../../assets/Images/yi.svg";
import ShimmerUI from "./ShimmerUI";
import "./Monitor.module.css";
// Register the necessary Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

const DataDashboard = () => {
  const [data, setData] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const navigate = useNavigate(); // Initialize the useNavigate hook

  useEffect(() => {
    // Fetch data for charts
    axios
      .get(
        "/data"
      )
      .then((response) => setData(response.data))
      .catch((error) => {
        console.error("Error fetching data:", error);
        console.error("Error details:", error.message);
      });

    // Fetch data for analytics
    axios
      .get(
        "/analytics"
      )
      .then((response) => setAnalytics(response.data))
      .catch((error) => console.error("Error fetching analytics:", error));
  }, []);

  console.log(analytics);
  console.log(data);

  if (!data || !analytics) {
    return <ShimmerUI />;
  }
  // Find the specific entry and decrement its value
  data.pieChartDataNamesCount = data.pieChartDataNamesCount.map((item) =>
    item.name === "ChatCohere" ? { ...item, value: item.value - 1 } : item
  );
  const {
    pieChartDataStatus,
    pieChartDataNamesCount,
    pieChartDataRunTypesCount,
    tableData,
  } = data;

  console.log(
    pieChartDataStatus,
    pieChartDataNamesCount,
    pieChartDataRunTypesCount,
    tableData
  );

  const {
    averageTokensPerQuestion,
    averageCostPerQuestion,
    averageLatencyPerQuestion,
    totalCost,
    totalQuestions,
  } = analytics;

  // Modify tableData to include necessary fields and handle nested JSON objects
  const modified_tableData = tableData.map((row) => ({
    ...row,
    Inputs: JSON.stringify(row.Inputs, null, 2),
    outputs: JSON.stringify(row.outputs, null, 2),
  }));
  console.log(modified_tableData);

  return (
    <Grid
      container
      spacing={3}
      style={{ padding: "20px", backgroundColor: "#f5f5f5" }}
    >
      {/* Header with Logo */}
      <Grid item xs={12}>
        <Box
          display="flex"
          justifyContent="space-between"
          alignItems="center"
          style={{ marginBottom: "20px" }}
        >
          <img
            alt="InfoBell Logo"
            src={infobellImg}
            style={{ height: "3rem" }}
          />
          <img
            alt="Back Logo"
            src={backLogo}
            style={{ height: "2rem", cursor: "pointer" }}
            onClick={() => navigate("/")} // Navigate to home page on click
          />
        </Box>
      </Grid>

      {/* Analytics Cards */}
      <Grid item xs={12} md={2.4}>
        <Card
          sx={{
            height: "100%",
            backgroundColor: "#FFFFFFA8",
            boxShadow: "rgba(0, 0, 0, 0.24) 0px 3px 8px", // rgba(0, 0, 0, 0.05) is equivalent to #0000000D
          }}
        >
          <CardContent>
            <div
              style={{
                backgroundColor: "#DBE4F9",
                width: "4rem",
                height: "4rem",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                overflow: "hidden",
                borderRadius: "12px",
                marginBottom: "10px",
              }}
            >
              <img alt="InfoBell Logo" src={g1} style={{ height: "2.5rem" }} />
            </div>
            <p variant="h6" className="custom-typography">
              Average Tokens per Question
            </p>
            <p variant="h4" className="custom-typography2">
              {averageTokensPerQuestion.toFixed(2)}
            </p>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={2.4}>
        <Card
          sx={{
            height: "100%",
            backgroundColor: "#FFFFFFA8",
            boxShadow: "rgba(0, 0, 0, 0.24) 0px 3px 8px", // rgba(0, 0, 0, 0.05) is equivalent to #0000000D
          }}
        >
          <CardContent>
            <div
              style={{
                backgroundColor: "#DBE4F9",
                width: "4rem",
                height: "4rem",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                overflow: "hidden",
                borderRadius: "12px",
                marginBottom: "10px",
              }}
            >
              <img
                alt="InfoBell Logo"
                src={TotalUser}
                style={{ height: "2.5rem" }}
              />
            </div>
            <p variant="h6" className="custom-typography">
              Average Cost per Question
            </p>
            <p variant="h4" className="custom-typography2">
              ${averageCostPerQuestion.toFixed(4)}
            </p>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={2.4}>
        <Card
          sx={{
            height: "100%",
            backgroundColor: "#FFFFFFA8",
            boxShadow: "rgba(0, 0, 0, 0.24) 0px 3px 8px", // rgba(0, 0, 0, 0.05) is equivalent to #0000000D
          }}
        >
          <CardContent>
            <div
              style={{
                backgroundColor: "#DBE4F9",
                width: "4rem",
                height: "4rem",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                overflow: "hidden",
                borderRadius: "12px",
                marginBottom: "10px",
              }}
            >
              <img alt="InfoBell Logo" src={hi} style={{ height: "2.5rem" }} />
            </div>
            <p variant="h6" className="custom-typography">
              Total Cost
            </p>
            <p variant="h4" className="custom-typography2">
              ${totalCost}
            </p>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={2.4}>
        <Card
          sx={{
            height: "100%",
            backgroundColor: "#FFFFFFA8",
            boxShadow: "rgba(0, 0, 0, 0.24) 0px 3px 8px", // rgba(0, 0, 0, 0.05) is equivalent to #0000000D
          }}
        >
          <CardContent>
            <div
              style={{
                backgroundColor: "#DBE4F9",
                width: "4rem",
                height: "4rem",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                overflow: "hidden",
                borderRadius: "12px",
                marginBottom: "10px",
              }}
            >
              <img alt="InfoBell Logo" src={ri} style={{ height: "2.5rem" }} />
            </div>
            <p variant="h6" className="custom-typography">
              Average TTFT per Question
            </p>
            <p variant="h4" className="custom-typography2">
              {averageLatencyPerQuestion.toFixed(2)} ms
            </p>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={2.4}>
        <Card
          sx={{
            height: "100%",
            backgroundColor: "#FFFFFFA8",
            boxShadow: "rgba(0, 0, 0, 0.24) 0px 3px 8px", // rgba(0, 0, 0, 0.05) is equivalent to #0000000D
          }}
        >
          <CardContent>
            <div
              style={{
                backgroundColor: "#DBE4F9",
                width: "4rem",
                height: "4rem",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                overflow: "hidden",
                borderRadius: "12px",
                marginBottom: "10px",
              }}
            >
              <img alt="InfoBell Logo" src={yi} style={{ height: "2.5rem" }} />
            </div>
            <p variant="h6" className="custom-typography">
              Total Questions
            </p>
            <p variant="h4" className="custom-typography2">
              {totalQuestions}
            </p>
          </CardContent>
        </Card>
      </Grid>
      {/* <div className="custom-div">
        <Grid container spacing={2} direction="row" justifyContent="center">
          <Grid
            item
            xs={12}
            md={4}
            display="flex"
            flexDirection="column"
            alignItems="center"
          >
            <DoughnutChart
              percentages={pieChartDataStatus.map((item) => item.value)}
              labels={pieChartDataStatus.map((item) => item.name)}
            />
            <Typography
              variant="h6"
              className="custom-ChartText"
              style={{
                marginTop: "5px",
                marginBottom: "10px",
                fontWeight: "bold",
                textAlign: "center",
              }}
            >
              Status Distribution
            </Typography>
          </Grid>

          <Grid
            item
            xs={12}
            md={4}
            display="flex"
            flexDirection="column"
            alignItems="center"
          >
            <DoughnutChart
              percentages={pieChartDataNamesCount.map((item) => item.value)}
              labels={pieChartDataNamesCount.map((item) => item.name)}
            />
            <Typography
              className="custom-ChartText"
              variant="h6"
              style={{
                marginTop: "5px",
                marginBottom: "10px",
                fontWeight: "bold",
                textAlign: "center",
              }}
            >
              Names Count
            </Typography>
          </Grid>

          <Grid
            item
            xs={12}
            md={4}
            display="flex"
            flexDirection="column"
            alignItems="center"
          >
            <DoughnutChart
              percentages={pieChartDataRunTypesCount.map((item) => item.value)}
              labels={pieChartDataRunTypesCount.map((item) => item.name)}
            />
            <Typography
              variant="h6"
              className="custom-ChartText"
              style={{
                marginTop: "5px",
                marginBottom: "10px",
                fontWeight: "bold",
                textAlign: "center",
              }}
            >
              Run Types Count
            </Typography>
          </Grid>
        </Grid>
      </div> */}

      {/* Table */}
      <ColumnGroupingTable tableData={modified_tableData} />
    </Grid>
  );
};

export default DataDashboard;
