// import "./App.css";
import Chatbot from "react-chatbot-kit";
import config from "../../chatbot/config";
import ActionProvider from "../../chatbot/ActionProvider.jsx";
import MessageParser from "../../chatbot/MessageParser";

function Chat() {
  return (
    <div className="App">
      <Chatbot
        config={config}
        actionProvider={ActionProvider}
        messageParser={MessageParser}
        placeholderText={"enter your question here"}
      />
    </div>
  );
}

export default Chat;
