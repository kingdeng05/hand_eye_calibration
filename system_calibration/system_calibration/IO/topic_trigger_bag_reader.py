import rosbag

from .bag_reader import BagReader


class TopicTriggerBagReader(BagReader):
    def __init__(self, bag_name, trigger_topic: str, *topics):
        super().__init__(bag_name) 
        self._trigger_topic = trigger_topic
        self._topics = list(topics) 
        if len(self._topics) <= 0:
            raise ValueError("input topics have to be length >= 1")
        self._check_topics()

    def read(self, bag):
        with rosbag.Bag(self._name, 'r') as bag:
            trigger_time = None
            all_topics = [self._trigger_topic] + self._topics
            msgs = None 
            for topic, msg, _ in bag.read_messages(topics=all_topics):
                # Check if the current message is from the trigger topic
                if topic == self._trigger_topic:
                    trigger_time = msg.header.stamp
                    msgs = self._get_default_msgs()

                # If the trigger_time is set, process messages from other topics
                if trigger_time and topic in self._topics:
                    # print(f"found: {topic}")
                    msg_time = msg.header.stamp
                    if msg_time >= trigger_time:
                        msgs[self._topics.index(topic)] = (msg_time, msg) 

                if msgs and all([msg is not None for msg in msgs]):
                    yield msgs 

            if msgs is None:
                print(f"The bag may not contain {self._trigger_topic}, please double check!")

    def _list_topics(self):
        with rosbag.Bag(self._name, 'r') as bag:
            return bag.get_type_and_topic_info()[1].keys()

    def _check_topics(self):
        topics_available = self._list_topics()
        topics_missing = []
        for topic in [self._trigger_topic] + self._topics:
            if topic not in topics_available:
                topics_missing.append(topic)
        if len(topics_missing): 
            error_msg = f"\nFollowing topics are missing in the bag {self._name}:\n"
            for topic in topics_missing:
                error_msg += f"  {topic}\n" 
            raise RuntimeError(error_msg) 

    def _get_default_msgs(self):
        return [None] * len(self._topics)
    

    
    