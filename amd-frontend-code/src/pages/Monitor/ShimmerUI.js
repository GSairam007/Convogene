// ShimmerUI.js
import React from "react";
import "./Shimmer.css"; // Import the CSS file for styles
import CircularProgress from '@mui/material/CircularProgress';

const ShimmerUI = () => {
  return (
    <div className="loading-container" style={{
      color: "#5391F6"
    }}>
      <CircularProgress color="inherit" />
      <span className="loading-text" style={{
        marginTop: "10px",
      }}>Loading Data...</span>
    </div>
  );
};

export default ShimmerUI;
