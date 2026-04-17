// AdvancedShimmer.js
import React from "react";
import "./Shimmer.css";

const AdvancedShimmer = () => {
  return (
    <div className="shimmer-container">
      {/* Avatar Shimmer */}
      <div className="shimmer-block shimmer-circle">
        <div className="shimmer"></div>
      </div>

      {/* Title and Text Lines Shimmer */}

      <div className="shimmer-block shimmer-line">
        <div className="shimmer"></div>
      </div>
      <div className="shimmer-block shimmer-line">
        <div className="shimmer"></div>
      </div>
      <div className="shimmer-block shimmer-line short">
        <div className="shimmer"></div>
      </div>
    </div>
  );
};

export default AdvancedShimmer;
