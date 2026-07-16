from cicflowmeter.flow_session import FlowSession
from scapy.all import PcapReader, IP, TCP
import os

with open('dummy.pcap', 'wb') as f:
    f.write(b'')

# Write some actual packets
from scapy.utils import wrpcap
pkts = [IP(src="1.1.1.1", dst="2.2.2.2")/TCP(sport=100, dport=200)] * 5
wrpcap("dummy.pcap", pkts)

session = FlowSession(output_mode="csv", output="dummy.csv")
try:
    for pkt in PcapReader('dummy.pcap'):
        if pkt.haslayer("IP") and (pkt.haslayer("TCP") or pkt.haslayer("UDP")):
            session.process(pkt)
finally:
    session.flush_flows()

with open("dummy.csv", "r") as f:
    print(f.read())
