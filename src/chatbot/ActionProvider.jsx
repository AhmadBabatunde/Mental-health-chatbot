import React from 'react';
import axios from 'axios';
import Loader from "../components/loader";

class ActionProvider {
  constructor(
    createChatBotMessage,
    setStateFunc,
    createClientMessage,
    stateRef,
    createCustomMessage,
    ...rest
  ) {
    this.createChatBotMessage = createChatBotMessage;
    this.setState = setStateFunc;
    this.createClientMessage = createClientMessage;
    this.stateRef = stateRef;
    this.createCustomMessage = createCustomMessage;
  }

  // Function to add the loader message to the state
  loader = () => {
    console.log("in loader");
    const loadingMessage = this.createChatBotMessage(<Loader />, {
      option: { delay: 0 },
    });

    this.addMessageToState(loadingMessage);
  };

  // Function to remove the loader message from the state
  RemoveLoader = () => {
    console.log("removing loader");
    this.setState((prevState) => {
      // Only keep non-loading messages and remove widgets that are not set properly
      const updatedMessages = prevState.messages.filter(
        (msg) => typeof msg.message === "string" || msg.widget === "unknown"
      );
      return { ...prevState, messages: updatedMessages };
    });
  };

  handleAskQuestion = async (question) => {
    // Show the loader message
    this.loader();
    
    if (!question) {
      this.handleApiResponse("Please ask your question");
      return;
    }

    try {
      const url = "http://127.0.0.1:5001/chat"; // Your API URL
      const response = await axios.post(url, {
        question: question,
        user_id: localStorage.getItem("user_id"),
      });

      console.log("API response received:", response);

      if (response && response.data && response.data.response) {
        this.handleApiResponse(response.data.response);
      } else {
        console.error("Invalid response format:", response);
        this.handleApiResponse("Sorry, I couldn't understand that.");
      }
    } catch (error) {
      this.handleApiResponse("An error occurred. Please try again.");
      console.error("Error asking question:", error);
    }
  };

  handleApiResponse = (response) => {
    console.log("handleApiResponse called with:", response);

    // Remove the loader before adding the new response
    this.RemoveLoader();

    // If response is empty, set a fallback response
    if (!response || response.trim() === "") {
      console.error("Empty response received:", response);
      response = "Sorry, I didn't get that. Could you please rephrase?";
    }

    this.setState((prevState) => {
      console.log("Previous state:", prevState);

      // Create a new message from the response
      const responseMessage = this.createChatBotMessage(response);
      console.log("Created response message:", responseMessage);

      const newState = {
        ...prevState,
        messages: [...prevState.messages, responseMessage],
      };

      console.log("New state to be set:", newState);

      return newState;
    });
  };

  addMessageToState = (message) => {
    console.log("Adding message to state:", message);
    this.setState((prevState) => {
      const newState = {
        ...prevState,
        messages: [...prevState.messages, message],
      };
      console.log("New state after adding message:", newState);
      return newState;
    });
  };
}

export default ActionProvider;
