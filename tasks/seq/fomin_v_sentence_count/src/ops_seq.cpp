


bool pre_processing() {
    internal_order_test();
    // Получаем входную строку
    input_ = reinterpret_cast<char*>(taskData->inputs[0]);
    sentence_count = 0;
    return true;
  }

  bool validation() {
    internal_order_test();
    // Проверяем, что входные данные содержат строку
    return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
  }

  bool run() {
    internal_order_test();
    // Подсчитываем количество предложений
    for (int i = 0; input_[i] != '\0'; ++i) {
      if (ispunct(input_[i]) && (input_[i] == '.' || input_[i] == '!' || input_[i] == '?')) {
        sentence_count++;
      }
    }
    return true;
  }

  bool post_processing() {
    internal_order_test();
    // Записываем результат в выходные данные
    reinterpret_cast<int*>(taskData->outputs[0])[0] = sentence_count;
    return true;