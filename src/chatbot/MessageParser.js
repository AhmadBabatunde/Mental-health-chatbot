class MessageParser {
  constructor(actionProvider) {
    this.actionProvider = actionProvider;
  }

  parse(message) {
    // console.log(message);

    this.actionProvider.handleAskQuestion(message);
  }
}

export default MessageParser;
