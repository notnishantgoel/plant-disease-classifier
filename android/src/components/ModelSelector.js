import React from 'react'
import {
  View, Text, ScrollView, TouchableOpacity, StyleSheet,
} from 'react-native'

export default function ModelSelector({ models, selected, onSelect }) {
  return (
    <View style={styles.container}>
      <Text style={styles.label}>SELECT MODEL</Text>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.row}>
        {models.map((m) => {
          const active = selected?.id === m.id
          return (
            <TouchableOpacity
              key={m.id}
              style={[styles.card, active && styles.cardActive]}
              onPress={() => onSelect(m)}
              activeOpacity={0.75}
            >
              <Text style={styles.icon}>{m.icon}</Text>
              <Text style={[styles.name, active && styles.nameActive]}>{m.shortName}</Text>
              <Text style={styles.acc}>{m.accuracy}</Text>
            </TouchableOpacity>
          )
        })}
      </ScrollView>
      {selected && (
        <Text style={styles.desc}>{selected.description}</Text>
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  container: { marginBottom: 20 },
  label: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1,
    color: '#64748b',
    marginBottom: 10,
  },
  row: { gap: 10, paddingBottom: 2 },
  card: {
    alignItems: 'center',
    padding: 12,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#e2e8f0',
    backgroundColor: '#fff',
    minWidth: 80,
    gap: 4,
  },
  cardActive: {
    borderColor: '#40916c',
    backgroundColor: '#f0fdf4',
  },
  icon: { fontSize: 22 },
  name: { fontSize: 11, fontWeight: '600', color: '#1e293b', textAlign: 'center' },
  nameActive: { color: '#2d6a4f' },
  acc: { fontSize: 10, color: '#40916c', fontWeight: '700' },
  desc: {
    marginTop: 10,
    fontSize: 12.5,
    color: '#64748b',
    lineHeight: 18,
    backgroundColor: '#f8fafc',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
})
