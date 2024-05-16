// eslint-disable-next-line no-unused-vars
import React from "react";
import bgimg from "../assets/bg-art.jpg";

// eslint-disable-next-line react/prop-types
export default function BackgroundImage({ children }) {
  return (
    <>
      <div className="hidden md:flex justify-center relative overflow-hidden">
        <img
          src={bgimg}
          alt="sectionimage"
          className=" w-screen h-full object-cover z-0"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-bg-primary via-[#00000000] via-50% to-bg-primary z-10">
          {children}
        </div>
      </div>
    </>
  );
}
