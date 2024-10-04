import React from "react";

const BotAvatar = () => {
  return (
    <div
      style={{
        width: "40px",
        height: "40px",
        borderRadius: "50%",
        overflow: "hidden", // This ensures the image stays within the rounded container
      }}
    >
      <img
        src="https://t4.ftcdn.net/jpg/04/46/38/69/240_F_446386956_DiOrdcxDFWKWFuzVUCugstxz0zOGMHnA.jpg"
        alt="bot image"
        style={{
          width: "100%", // Ensure the image fills the container width
          height: "100%", // Ensure the image fills the container height
          objectFit: "cover", // Maintain aspect ratio and cover the entire container
        }}
      />
    </div>
  );
};

export default BotAvatar;
