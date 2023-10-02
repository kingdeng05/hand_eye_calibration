import rosbag

from .bag_reader import BagReader
from ..utils import convert_to_unix_ns


class TopicTriggerBagReader(BagReader):
    def __init__(self, bag_name, *topics):
        super().__init__(bag_name) 
        self._topics = list(topics) 
        self._trigger_topic = self._topics[0] 
        if len(self._topics) <= 0:
            raise ValueError("input topics have to be length >= 1")
        self._check_topics()

    def read(self):
        with rosbag.Bag(self._name, 'r') as bag:
            trigger_time = None
            msgs = None 
            for topic, msg, t_msg_received in bag.read_messages(topics=self._topics):
                # Check if the current message is from the trigger topic
                if topic == self._trigger_topic:
                    trigger_time = msg.header.stamp.to_sec()
                    msgs = self._get_default_msgs()

                # If the trigger_time is set, process messages from other topics
                if trigger_time and topic in self._topics:
                    if hasattr(msg, "header"):
                        msg_time = msg.header.stamp.to_sec()
                    else:
                        msg_time = t_msg_received.to_sec()
                    topic_idx = self._topics.index(topic)
                    if msg_time >= trigger_time and msgs[topic_idx] is None:
                        msgs[self._topics.index(topic)] = (convert_to_unix_ns(msg_time), msg) 

                if msgs and all(msgs):
                    yield msgs 
                    # clear the msgs
                    msgs = self._get_default_msgs() 

    def _list_topics(self):
        with rosbag.Bag(self._name, 'r') as bag:
            return bag.get_type_and_topic_info()[1].keys()

    def _check_topics(self):
        topics_available = self._list_topics()
        topics_missing = []
        for topic in self._topics:
            if topic not in topics_available:
                topics_missing.append(topic)
        if len(topics_missing): 
            error_msg = f"\nFollowing topics are missing in the bag {self._name}:\n"
            for topic in topics_missing:
                error_msg += f"  {topic}\n" 
            raise RuntimeError(error_msg) 

    def _get_default_msgs(self):
        return [None] * len(self._topics)
    
    