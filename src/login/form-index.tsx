import React, { useState } from "react";
import "./index.css"; // Import a CSS file for styling

const TherapyForm = ({ onSubmitForm }) => {
  const [formData, setFormData] = useState({
    fullname:"",
    therapyReason: "",
    beenToTherapy: "",
    age: "",
    stressLevel: 1,
    occupation: "",
    therapyGoals: "",
    significantEvents: "",
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    // Create the string based on the form data
    const formattedString = `
      My name is {${formData.fullname}},
      What brings you to therapy now? {${formData.therapyReason}},
      Have you been to therapy before? {${formData.beenToTherapy}},
      How old are you? {${formData.age}},
      On a scale of 1 to 10, how would you rate your current stress level? {${formData.stressLevel}},
      What is your current occupation? {${formData.occupation}},
      What are your goals for therapy? {${formData.therapyGoals}},
      What significant life events have you experienced recently? {${formData.significantEvents}}`;

    // Pass the string to the parent component or chatbot
    onSubmitForm(formattedString);

    // Optionally log it for debugging
    console.log(formattedString);
  };

  return (
    <div className="form-container">
      <form onSubmit={handleSubmit} className="therapy-form">
        <h2>Therapy Intake Form</h2>

        <div className="form-group">
          <label>What is your name?</label>
          <input
            type="text"
            name="therapyReason"
            value={formData.fullname}
            onChange={handleChange}
            placeholder="Enter your full name"
            required
          />
        </div>
        
        <div className="form-group">
          <label>What brings you to therapy now?</label>
          <input
            type="text"
            name="therapyReason"
            value={formData.therapyReason}
            onChange={handleChange}
            placeholder="Enter reason for therapy"
            required
          />
        </div>

        <div className="form-group">
          <label>Have you been to therapy before?</label>
          <select
            name="beenToTherapy"
            value={formData.beenToTherapy}
            onChange={handleChange}
            required
          >
            <option value="">Select an option</option>
            <option value="yes">Yes</option>
            <option value="no">No</option>
          </select>
        </div>

        <div className="form-group">
          <label>How old are you?</label>
          <input
            type="number"
            name="age"
            value={formData.age}
            onChange={handleChange}
            placeholder="Enter your age"
            required
          />
        </div>

        <div className="form-group">
          <label>
            On a scale of 1 to 10, how would you rate your current stress level?
          </label>
          <input
            type="range"
            name="stressLevel"
            min="1"
            max="10"
            value={formData.stressLevel}
            onChange={handleChange}
          />
          <span>{formData.stressLevel}</span>
        </div>

        <div className="form-group">
          <label>What is your current occupation?</label>
          <input
            type="text"
            name="occupation"
            value={formData.occupation}
            onChange={handleChange}
            placeholder="Enter your occupation"
            required
          />
        </div>

        <div className="form-group">
          <label>What are your goals for therapy?</label>
          <input
            type="text"
            name="therapyGoals"
            value={formData.therapyGoals}
            onChange={handleChange}
            placeholder="Enter your therapy goals"
            required
          />
        </div>

        <div className="form-group">
          <label>
            What significant life event have you experienced recently?
          </label>
          <input
            type="text"
            name="significantEvents"
            value={formData.significantEvents}
            onChange={handleChange}
            placeholder="Describe significant life events"
            required
          />
        </div>

        <div className="form-buttons">
          <button type="button" className="skip-btn">
            Skip
          </button>
          <button type="submit" className="submit-btn">
            Submit
          </button>
        </div>
      </form>
    </div>
  );
};

export default TherapyForm;
