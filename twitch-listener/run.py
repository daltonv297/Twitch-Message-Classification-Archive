from twitch_listener import listener

# Connect to Twitch
bot = listener.connect_twitch('mrbelms', 
                             'oauth:6esuhj8iq1gsjkgfukv1o94if5p7de', 
                             'ttv_listener')

# List of channels to connect to
#channels_to_listen_to = ['xQc', 'KaiCenat', 'shroud', 'Amouranth', 'AdinRoss', 'loltyler1', 'Wirtual', 'Pestily', 'HasanAbi', 'moistcr1tikal', 'erobb221', 'tarik', 'pokimane', 'sodapoppin', 'NICKMERCS', 'summit1g', 'Loserfruit', 'LIRIK', 'Anomaly', 'Fextralife', 'Tfue', 'Castro_1021', 'Heelmike', 'tommyinnit']
channels_to_listen_to = ['shroud', 'moistcr1tikal']
# Scrape live chat data into raw log files. (Duration is seconds)
bot.listen(channels_to_listen_to, duration = 10) 

# Convert log files into .CSV format
bot.parse_logs(timestamp = True)

# Generate adjacency matrix
bot.adj_matrix(weighted = True, matrix_name = "streamer_network.csv")