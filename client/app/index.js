import { View, Text } from 'react-native'
import { Link } from 'expo-router';
import React from 'react'

const index = () => {
    return (
        <View className="flex-1 items-center justify-center bg-black">
            <Text>index!</Text>
            <Link href="/home">Continue
            </Link>
        </View>
    )
}

export default index